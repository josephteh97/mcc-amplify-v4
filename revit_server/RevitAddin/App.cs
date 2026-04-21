// =============================================================================
// RevitModelBuilderAddin — Revit 2023 Add-in
// =============================================================================
//
// Architecture
// ------------
// The Revit API can only be called on Revit's main UI thread.  A standalone
// TCP server runs on a background thread and receives JSON requests from the
// Python backend.  Revit API work is marshalled to the main thread via two
// ExternalEvent + IExternalEventHandler pairs:
//
//   1. BuildHandler  — monolithic batch build  (POST /build-model)
//   2. CommandHandler — single-step commands   (session/* routes)
//
// Endpoints — Batch (legacy, fully backward-compatible)
// -------------------------------------------------------
//   GET  /health
//   POST /build-model   → { job_id, transaction_json }
//                          Returns binary .rvt on success.
//
// Endpoints — Stateful Session (new, for MCP agent workflow)
// ----------------------------------------------------------
//   POST /session/new
//       → { session_id, levels: [...], message }
//
//   POST /session/{id}/load-family
//       body: { rfa_path, windows_rfa_path? }
//       → { family_name, types: [{type_name, ...}], already_loaded }
//
//   GET  /session/{id}/families
//       → { families: [{ name, category, types: [...] }] }
//
//   POST /session/{id}/place
//       body: { family_name, type_name, x_mm, y_mm, z_mm,
//               level, top_level?, parameters?: { name: value } }
//       → { element_id, placed: { category, family_name, type_name, ... } }
//
//   POST /session/{id}/set-param
//       body: { element_id, parameter_name, value, value_type? }
//             value_type: "mm" (default, converted) | "raw" | "string" | "int"
//       → { ok, parameter_name, new_value }
//
//   POST /session/{id}/get-params
//       body: { element_id }
//       → { element_id, count, parameters: [{name, storage_type, is_read_only, value_mm?}] }
//
//   POST /session/{id}/wall-join
//       (no body required)
//       → { ok, walls_total, walls_joined, warnings }
//
//   POST /session/{id}/export-view
//       (no body required)
//       → binary PNG bytes of the first floor plan view (150 DPI, 2048 px)
//
//   GET  /session/{id}/state
//       → { session_id, levels, families, placed_elements, warnings }
//
//   POST /session/{id}/export
//       body: { keep_open?: bool }   (default false — closes session)
//       → binary .rvt bytes
//
//   POST /session/{id}/close
//       → { ok, message }
//
//   POST /session/{id}/query-elements
//       body: { "element_ids": [int]?, "category": string? }   // either filter; both optional → all
//       → { elements: [ { id, category, family, type, bbox: { min, max }, location } ],
//           truncated?: true }   // truncated=true iff an unfiltered/category scan hit QUERY_ELEMENTS_MAX_SCAN
//         Distances in millimetres.  Used by the self-healing agent's clash resolver.
//
// Session lifecycle
// -----------------
//   new-session → load-family* → place* → set-param* → export → (auto-close)
//   Revit document stays open between steps within one session.
//   Max one concurrent command on the Revit thread; HTTP threads queue via
//   SemaphoreSlim.
//
// JSON Schema (transaction_json for /build-model — unchanged from v1)
// -------------------------------------------------------------------
//   levels, grids, columns, walls, floors  — all lengths in mm
//
// Deployment
// ----------
//   1. Build → bin\Release\net48\RevitModelBuilderAddin.dll
//   2. Copy RevitModelBuilder.addin + DLL to
//        C:\ProgramData\Autodesk\Revit\Addins\2023\
//   3. Start Revit 2023 → add-in loads automatically on port 5000
// =============================================================================

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

using Autodesk.Revit.ApplicationServices;
using Autodesk.Revit.Attributes;
using Autodesk.Revit.DB;
using Autodesk.Revit.DB.Structure;
using Autodesk.Revit.UI;

using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace RevitModelBuilderAddin
{
    // =========================================================================
    // Entry point: IExternalApplication
    // =========================================================================

    [Transaction(TransactionMode.Manual)]
    [Regeneration(RegenerationOption.Manual)]
    public class App : IExternalApplication
    {
        // ── Batch build (existing) ────────────────────────────────────────────
        private BuildHandler   _buildHandler;
        private ExternalEvent  _buildEvent;

        // ── Step-by-step session commands (new) ──────────────────────────────
        private CommandHandler    _cmdHandler;
        private ExternalEvent     _cmdEvent;
        private SemaphoreSlim     _cmdLock;     // serialises Revit thread access
        private RevitSessionManager _sessions;

        // ── TCP ───────────────────────────────────────────────────────────────
        private TcpListener             _tcpListener;
        private CancellationTokenSource _cts;

        // ── Warning capture (batch build) ─────────────────────────────────────
        // Populated by OnFailuresProcessing for any warning that escapes a
        // transaction's IFailuresPreprocessor.  Reset at the start of each
        // /build-model and consumed in HandleBuildModel's merge block.
        private readonly List<(string Text, List<int> ElementIds)> _sessionWarningDetails
            = new List<(string, List<int>)>();
        private readonly object _warningsLock = new object();

        private const int    PORT                    = 5000;
        private const double FEET_TO_MM              = 304.8;
        // Maximum elements returned from an unfiltered /query-elements scan.
        // Callers wanting more must pass explicit element_ids.
        private const int    QUERY_ELEMENTS_MAX_SCAN = 5000;

        // ── Startup ────────────────────────────────────────────────────────────

        public Result OnStartup(UIControlledApplication application)
        {
            string logPath = @"C:\RevitOutput\addin_startup.log";
            try { Directory.CreateDirectory(@"C:\RevitOutput"); } catch { }

            try
            {
                File.WriteAllText(logPath, $"[{DateTime.Now}] OnStartup called.\r\n");

                // Batch handler (existing)
                _buildHandler = new BuildHandler();
                _buildEvent   = ExternalEvent.Create(_buildHandler);

                // Step-command handler (new)
                _cmdHandler = new CommandHandler();
                _cmdEvent   = ExternalEvent.Create(_cmdHandler);
                _cmdLock    = new SemaphoreSlim(1, 1);
                _sessions   = new RevitSessionManager();

                // Warning capture for batch builds
                application.ControlledApplication.FailuresProcessing +=
                    OnFailuresProcessing;

                File.AppendAllText(logPath, $"[{DateTime.Now}] ExternalEvents created.\r\n");

                _cts         = new CancellationTokenSource();
                _tcpListener = new TcpListener(IPAddress.Any, PORT);
                _tcpListener.Start();

                File.AppendAllText(logPath,
                    $"[{DateTime.Now}] TcpListener started on 0.0.0.0:{PORT}\r\n");

                Task.Run(() => ListenLoop(_cts.Token));

                Console.WriteLine($"[RevitModelBuilderAddin] TCP server started on port {PORT}.");
                return Result.Succeeded;
            }
            catch (Exception ex)
            {
                string msg = $"[{DateTime.Now}] FAILED: {ex.GetType().Name}: {ex.Message}\r\n{ex.StackTrace}\r\n";
                try { File.AppendAllText(logPath, msg); } catch { }
                return Result.Failed;
            }
        }

        public Result OnShutdown(UIControlledApplication application)
        {
            try { application.ControlledApplication.FailuresProcessing -= OnFailuresProcessing; } catch { }
            try { _sessions?.CloseAll(); } catch { }
            try { _cts?.Cancel(); }        catch { }
            try { _tcpListener?.Stop(); }  catch { }
            return Result.Succeeded;
        }

        // ── FailuresProcessing (batch build warning capture) ──────────────────

        private void OnFailuresProcessing(
            object sender, Autodesk.Revit.DB.Events.FailuresProcessingEventArgs e)
        {
            // Application-level safety net: captures and dismisses any Revit
            // warnings that slipped past a transaction's IFailuresPreprocessor.
            // This prevents Revit from showing blocking yellow-triangle dialogs
            // during automated (non-interactive) pipeline runs.
            try
            {
                var fa = e.GetFailuresAccessor();
                bool anyHandled = false;
                foreach (var msg in fa.GetFailureMessages().ToList())
                {
                    string text = msg.GetDescriptionText();
                    if (!string.IsNullOrWhiteSpace(text))
                    {
                        var elemIds = msg.GetFailingElementIds()
                                         .Select(id => id.IntegerValue).ToList();
                        lock (_warningsLock)
                        {
                            _sessionWarningDetails.Add((text, elemIds));
                        }
                        Console.WriteLine($"[RevitWarning] {text}");
                    }

                    if (msg.GetSeverity() == FailureSeverity.Warning)
                    {
                        try { fa.DeleteWarning(msg); anyHandled = true; } catch { }
                    }
                    else if (FailureHelpers.IsJoinError(text))
                    {
                        // Safety net: a join error slipped past the transaction's
                        // IFailuresPreprocessor (WarningCollector) and would
                        // otherwise pop a modal dialog that hangs the build.
                        var applied = FailureHelpers.TryApplyJoinResolution(fa, msg);
                        if (applied != null)
                        {
                            anyHandled = true;
                            Console.WriteLine($"[JOIN-RESOLVED-APPLEVEL:{applied}] {text}");
                        }
                    }
                }
                if (anyHandled)
                    e.SetProcessingResult(FailureProcessingResult.Continue);
            }
            catch { }
        }

        // ── TCP accept loop ────────────────────────────────────────────────────

        private async void ListenLoop(CancellationToken ct)
        {
            while (!ct.IsCancellationRequested)
            {
                TcpClient client;
                try   { client = await _tcpListener.AcceptTcpClientAsync(); }
                catch { break; }
                _ = Task.Run(() => HandleClient(client));
            }
        }

        // ── Per-connection request handler ─────────────────────────────────────

        private void HandleClient(TcpClient client)
        {
            using (client)
            {
                NetworkStream stream = null;
                try
                {
                    stream = client.GetStream();
                    stream.ReadTimeout  = 15_000;
                    stream.WriteTimeout = 180_000;

                    string requestLine = ReadLine(stream);
                    if (string.IsNullOrWhiteSpace(requestLine)) return;

                    var parts  = requestLine.Split(' ');
                    string method = parts.Length > 0 ? parts[0] : "";
                    string path   = parts.Length > 1 ? parts[1] : "/";

                    var headers = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
                    string hdrLine;
                    while (!string.IsNullOrEmpty(hdrLine = ReadLine(stream)))
                    {
                        int colon = hdrLine.IndexOf(':');
                        if (colon > 0)
                            headers[hdrLine.Substring(0, colon).Trim()] =
                                hdrLine.Substring(colon + 1).Trim();
                    }

                    string body = "";
                    if (headers.TryGetValue("Content-Length", out string clStr) &&
                        int.TryParse(clStr, out int contentLength) && contentLength > 0)
                    {
                        byte[] bodyBytes = new byte[contentLength];
                        int read = 0;
                        while (read < contentLength)
                            read += stream.Read(bodyBytes, read, contentLength - read);
                        body = Encoding.UTF8.GetString(bodyBytes);
                    }

                    // ── Router ────────────────────────────────────────────────

                    if (method == "GET" && path == "/health")
                    {
                        WriteText(stream, 200, "Revit Model Builder ready");
                    }
                    else if (method == "POST" && path == "/build-model")
                    {
                        HandleBuildModel(stream, body);
                    }
                    else if (method == "POST" && path == "/session/new")
                    {
                        HandleNewSession(stream, body);
                    }
                    else if (method == "POST" && IsSessionPath(path, out string sid, out string cmd))
                    {
                        switch (cmd)
                        {
                            case "load-family":    HandleLoadFamily(stream, sid, body);    break;
                            case "place":          HandlePlaceInstance(stream, sid, body); break;
                            case "set-param":      HandleSetParam(stream, sid, body);      break;
                            case "get-params":     HandleGetParams(stream, sid, body);     break;
                            case "wall-join":      HandleWallJoinAll(stream, sid);         break;
                            case "export-view":    HandleExportView(stream, sid);          break;
                            case "export":         HandleExportSession(stream, sid, body); break;
                            case "close":          HandleCloseSession(stream, sid);        break;
                            case "query-elements": HandleQueryElements(stream, sid, body); break;
                            default:               WriteJson(stream, 404, new { error = $"Unknown command: {cmd}" }); break;
                        }
                    }
                    else if (method == "GET" && IsSessionPath(path, out string gSid, out string gCmd))
                    {
                        switch (gCmd)
                        {
                            case "families": HandleListFamilies(stream, gSid);  break;
                            case "state":    HandleQueryState(stream, gSid);    break;
                            default:         WriteJson(stream, 404, new { error = $"Unknown command: {gCmd}" }); break;
                        }
                    }
                    else if (method == "GET" && path.StartsWith("/list-rfa"))
                    {
                        HandleListRfa(stream, path);
                    }
                    else
                    {
                        WriteText(stream, 404, "Not Found");
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[RevitModelBuilderAddin] Client error: {ex.Message}");
                    try { if (stream != null) WriteText(stream, 500, ex.Message); } catch { }
                }
            }
        }

        // Parses /session/{id}/{cmd}  → true when matched
        private static bool IsSessionPath(string path, out string sessionId, out string command)
        {
            sessionId = command = null;
            var parts = path.TrimStart('/').Split('/');
            if (parts.Length < 3 || parts[0] != "session") return false;
            sessionId = parts[1];
            command   = parts[2];
            return true;
        }

        // =========================================================================
        // Handler — Batch build (POST /build-model)  — UNCHANGED from v1
        // =========================================================================

        private void HandleBuildModel(NetworkStream stream, string requestBody)
        {
            var buildReq = JsonConvert.DeserializeObject<BuildRequest>(requestBody)
                           ?? throw new Exception("Could not deserialise build request.");

            string outputDir  = @"C:\RevitOutput";
            Directory.CreateDirectory(outputDir);
            string rvtName    = RvtFileName(buildReq.PdfFilename, buildReq.JobId);
            string outputPath = Path.Combine(outputDir, rvtName);

            lock (_warningsLock)
            {
                _sessionWarningDetails.Clear();
            }

            // If a previous event is still queued (e.g. left by a timed-out request or
            // a Revit dialog that blocked execution), wait up to 20 s for it to clear
            // before raising a new one.  Without this, rapid Python retries all pile up
            // against the already-pending event and all fail immediately.
            const int pendingPollMs = 300;
            const int pendingMaxMs  = 20_000;
            int pendingWaited = 0;
            while (_buildEvent.IsPending && pendingWaited < pendingMaxMs)
            {
                Thread.Sleep(pendingPollMs);
                pendingWaited += pendingPollMs;
            }
            if (_buildEvent.IsPending)
                throw new Exception(
                    "ExternalEvent still Pending after 20 s — Revit is likely showing a modal dialog. " +
                    "Dismiss any open dialogs in Revit and retry.");

            _buildHandler.Prepare(buildReq.TransactionJson, outputPath);
            ExternalEventRequest raised = _buildEvent.Raise();

            if (raised != ExternalEventRequest.Accepted)
                throw new Exception($"ExternalEvent rejected (status: {raised}). Revit may be busy.");

            if (!_buildHandler.Done.Wait(TimeSpan.FromSeconds(120)))
                throw new TimeoutException("Revit model build timed out after 120 s.");

            if (_buildHandler.BuildError != null)
                throw _buildHandler.BuildError;

            byte[] rvtBytes = File.ReadAllBytes(outputPath);

            // Merge across the two capture paths — OnFailuresProcessing
            // (session-scope) and WarningCollector (per-build) — since a
            // warning escaping the preprocessor is also seen by the safety net.
            // Element IDs from duplicate-text entries are unioned so bulk
            // failures (e.g. 74 join errors, same text, different IDs) retain
            // every affected element for the healing agent.
            var orderedTexts = new List<string>();
            var idsByText    = new Dictionary<string, HashSet<int>>();
            void AddEntry(string text, List<int> ids)
            {
                if (!idsByText.TryGetValue(text, out var set))
                {
                    set = new HashSet<int>();
                    idsByText[text] = set;
                    orderedTexts.Add(text);
                }
                foreach (var id in ids) set.Add(id);
            }
            lock (_warningsLock)
            {
                foreach (var d in _sessionWarningDetails) AddEntry(d.Text, d.ElementIds);
            }
            foreach (var d in _buildHandler.ResultWarningDetails) AddEntry(d.Text, d.ElementIds);

            var warningsV2 = orderedTexts.Select(text => new {
                text        = text,
                element_ids = idsByText[text].OrderBy(i => i).ToList(),
            }).ToList();

            string warningsJson = JsonConvert.SerializeObject(warningsV2);
            string familiesJson = JsonConvert.SerializeObject(_buildHandler.ResultFamilies);

            string respHeader =
                $"HTTP/1.1 200 OK\r\n" +
                $"Content-Type: application/octet-stream\r\n" +
                $"Content-Length: {rvtBytes.Length}\r\n" +
                $"Content-Disposition: attachment; filename={rvtName}\r\n" +
                $"X-Revit-Warnings: {warningsJson}\r\n" +
                $"X-Revit-Warnings-Version: 2\r\n" +
                $"X-Revit-Families: {familiesJson}\r\n" +
                $"Connection: close\r\n\r\n";

            byte[] headerBytes = Encoding.UTF8.GetBytes(respHeader);
            stream.Write(headerBytes, 0, headerBytes.Length);
            stream.Write(rvtBytes, 0, rvtBytes.Length);
            Console.WriteLine($"[RevitModelBuilderAddin] Built: {buildReq.JobId} ({rvtBytes.Length:N0} bytes)");
        }

        // =========================================================================
        // Step command — dispatch helper
        // Runs a lambda on the Revit UI thread; HTTP thread blocks until done.
        // Only one command runs at a time (serialised by _cmdLock).
        // =========================================================================

        private object RunOnRevitThread(Func<UIApplication, object> work, int timeoutMs = 60_000)
        {
            _cmdLock.Wait();
            try
            {
                _cmdHandler.SetCommand(work);
                var raised = _cmdEvent.Raise();
                if (raised != ExternalEventRequest.Accepted)
                    throw new Exception($"ExternalEvent rejected (status: {raised}). Revit may be busy.");
                return _cmdHandler.WaitResult(timeoutMs);
            }
            finally
            {
                _cmdLock.Release();
            }
        }

        // =========================================================================
        // Handler — POST /session/new
        // =========================================================================

        private void HandleNewSession(NetworkStream stream, string body)
        {
            try
            {
                var req = string.IsNullOrWhiteSpace(body)
                    ? new JObject()
                    : JObject.Parse(body);

                var result = (JObject)RunOnRevitThread(uiapp =>
                {
                    string template = ModelBuilder.FindTemplate();
                    if (template == null)
                        throw new Exception(
                            @"No .rte template found in C:\ProgramData\Autodesk\RVT 2023\Templates\. " +
                            "Reinstall Revit templates.");

                    Document doc = uiapp.Application.NewProjectDocument(template);

                    // Enumerate levels created by the template
                    var levels = new FilteredElementCollector(doc)
                        .OfClass(typeof(Level))
                        .Cast<Level>()
                        .OrderBy(l => l.Elevation)
                        .Select(l => new JObject {
                            ["name"]         = l.Name,
                            ["elevation_mm"] = Math.Round(l.Elevation * 304.8, 1),
                        })
                        .ToList();

                    string sessionId = _sessions.Create(doc, template);

                    Log($"Session {sessionId} created from template: {template}");
                    return new JObject {
                        ["session_id"] = sessionId,
                        ["message"]    = "Session created — document open and ready",
                        ["template"]   = Path.GetFileName(template),
                        ["levels"]     = new JArray(levels),
                    };
                });

                WriteJson(stream, 200, result);
            }
            catch (Exception ex)
            {
                WriteJson(stream, 500, new { error = ex.Message });
            }
        }

        // =========================================================================
        // Handler — POST /session/{id}/load-family
        // Body: { "rfa_path": "C:\\...\\M_Concrete-Rectangular-Column.rfa",
        //         "windows_rfa_path": "..." }   (optional alias)
        // =========================================================================

        private void HandleLoadFamily(NetworkStream stream, string sessionId, string body)
        {
            var state = _sessions.Get(sessionId);
            if (state == null) { WriteJson(stream, 404, new { error = "Session not found" }); return; }

            try
            {
                var req = JObject.Parse(body);
                string rfaPath = req["rfa_path"]?.ToString()
                              ?? req["windows_rfa_path"]?.ToString()
                              ?? throw new Exception("rfa_path is required");

                var result = (JObject)RunOnRevitThread(uiapp =>
                {
                    var doc = state.Document;

                    // Already loaded?
                    string familyName = Path.GetFileNameWithoutExtension(rfaPath);
                    var existing = new FilteredElementCollector(doc)
                        .OfClass(typeof(Family))
                        .Cast<Family>()
                        .FirstOrDefault(f => string.Equals(f.Name, familyName,
                                                           StringComparison.OrdinalIgnoreCase));
                    if (existing != null)
                    {
                        var existingTypes = BuildTypeArray(doc, existing);
                        return new JObject {
                            ["family_name"]    = existing.Name,
                            ["already_loaded"] = true,
                            ["types"]          = existingTypes,
                        };
                    }

                    // Load from disk
                    if (!File.Exists(rfaPath))
                        throw new Exception($"RFA file not found on this machine: {rfaPath}");

                    Family loaded = null;
                    var loadWc = new WarningCollector(@"C:\RevitOutput\session_log.txt");
                    using (var t = new Transaction(doc, $"Load {familyName}"))
                    {
                        var fo = t.GetFailureHandlingOptions();
                        fo.SetFailuresPreprocessor(loadWc);
                        t.SetFailureHandlingOptions(fo);
                        t.Start();
                        doc.LoadFamily(rfaPath, new OverwriteFamilyLoadOptions(), out loaded);
                        t.Commit();
                    }
                    loadWc.LogSummary($"Session {sessionId} load-family");

                    if (loaded == null)
                        throw new Exception($"LoadFamily returned null for {rfaPath}");

                    var types = BuildTypeArray(doc, loaded);
                    Log($"Session {sessionId}: loaded family '{loaded.Name}' ({types.Count} types)" +
                        (loadWc.Warnings.Count > 0 ? $" — {loadWc.Warnings.Count} warning(s)" : ""));
                    return new JObject {
                        ["family_name"]    = loaded.Name,
                        ["already_loaded"] = false,
                        ["types"]          = types,
                        ["warnings"]       = new JArray(loadWc.Warnings),
                    };
                });

                WriteJson(stream, 200, result);
            }
            catch (Exception ex)
            {
                WriteJson(stream, 500, new { error = ex.Message });
            }
        }

        private static JArray BuildTypeArray(Document doc, Family family)
        {
            var arr = new JArray();
            foreach (var symId in family.GetFamilySymbolIds())
            {
                if (!(doc.GetElement(symId) is FamilySymbol sym)) continue;
                var entry = new JObject {
                    ["type_name"]   = sym.Name,
                    ["element_id"]  = sym.Id.IntegerValue,
                };
                // Try to read common size parameters
                AppendParamMm(entry, sym, "b",             "b_mm");
                AppendParamMm(entry, sym, "h",             "h_mm");
                AppendParamMm(entry, sym, "Width",         "width_mm");
                AppendParamMm(entry, sym, "Depth",         "depth_mm");
                AppendParamMm(entry, sym, "Height",        "height_mm");
                AppendParamMm(entry, sym, "Diameter",      "diameter_mm");
                arr.Add(entry);
            }
            return arr;
        }

        private static void AppendParamMm(JObject target, FamilySymbol sym,
                                          string paramName, string jsonKey)
        {
            var p = sym.LookupParameter(paramName);
            if (p != null && p.StorageType == StorageType.Double)
                target[jsonKey] = Math.Round(p.AsDouble() * 304.8, 1);
        }

        // =========================================================================
        // Handler — GET /session/{id}/families
        // =========================================================================

        private void HandleListFamilies(NetworkStream stream, string sessionId)
        {
            var state = _sessions.Get(sessionId);
            if (state == null) { WriteJson(stream, 404, new { error = "Session not found" }); return; }

            try
            {
                var result = (JObject)RunOnRevitThread(uiapp =>
                {
                    var doc = state.Document;
                    var arr = new JArray();

                    foreach (Family fam in new FilteredElementCollector(doc)
                                              .OfClass(typeof(Family))
                                              .Cast<Family>()
                                              .OrderBy(f => f.Name))
                    {
                        var types = BuildTypeArray(doc, fam);
                        if (types.Count == 0) continue;
                        var catName = "";
                        try { catName = fam.FamilyCategory?.Name ?? ""; } catch { }
                        arr.Add(new JObject {
                            ["family_name"] = fam.Name,
                            ["category"]    = catName,
                            ["types"]       = types,
                        });
                    }

                    return new JObject {
                        ["session_id"] = sessionId,
                        ["families"]   = arr,
                    };
                });

                WriteJson(stream, 200, result);
            }
            catch (Exception ex)
            {
                WriteJson(stream, 500, new { error = ex.Message });
            }
        }

        // =========================================================================
        // Handler — GET /list-rfa?folder=<windows-path>
        // Lists all .rfa files in a folder on the Revit machine.
        // No Revit API required — runs directly on the TCP background thread.
        // Default folder: C:\MyDocuments\3. Revit Family Files
        // =========================================================================

        private const string DefaultUserFamilyFolder = @"C:\MyDocuments\3. Revit Family Files";

        private void HandleListRfa(NetworkStream stream, string path)
        {
            try
            {
                string folder = DefaultUserFamilyFolder;
                int qIdx = path.IndexOf('?');
                if (qIdx >= 0)
                {
                    string qs = path.Substring(qIdx + 1);
                    foreach (var pair in qs.Split('&'))
                    {
                        var kv = pair.Split(new[] { '=' }, 2);
                        if (kv.Length == 2 &&
                            string.Equals(kv[0], "folder", StringComparison.OrdinalIgnoreCase))
                        {
                            folder = Uri.UnescapeDataString(kv[1].Replace("+", " "));
                        }
                    }
                }

                if (!Directory.Exists(folder))
                {
                    WriteJson(stream, 200, new {
                        files   = new string[0],
                        count   = 0,
                        message = $"Folder not found on this machine: {folder}",
                    });
                    return;
                }

                string[] files = Directory.GetFiles(folder, "*.rfa", SearchOption.AllDirectories);
                WriteJson(stream, 200, new {
                    files  = files,
                    count  = files.Length,
                    folder = folder,
                });
            }
            catch (Exception ex)
            {
                WriteJson(stream, 500, new { error = ex.Message });
            }
        }

        // =========================================================================
        // Handler — POST /session/{id}/place
        //
        // Body: {
        //   "family_name": "M_Concrete-Rectangular-Column",
        //   "type_name":   "300×300mm",
        //   "x_mm": 6000, "y_mm": 0, "z_mm": 0,
        //   "level":     "Level 0",
        //   "top_level": "Level 1",          // optional; structural columns only
        //   "parameters": { "Mark": "C1" }   // optional instance parameters
        // }
        // =========================================================================

        private void HandlePlaceInstance(NetworkStream stream, string sessionId, string body)
        {
            var state = _sessions.Get(sessionId);
            if (state == null) { WriteJson(stream, 404, new { error = "Session not found" }); return; }

            try
            {
                var req      = JObject.Parse(body);
                string famName  = req["family_name"]?.ToString() ?? throw new Exception("family_name required");
                string typeName = req["type_name"]?.ToString()   ?? throw new Exception("type_name required");
                double xMm      = req["x_mm"]?.Value<double>()   ?? 0.0;
                double yMm      = req["y_mm"]?.Value<double>()   ?? 0.0;
                double zMm      = req["z_mm"]?.Value<double>()   ?? 0.0;
                string levelName    = req["level"]?.ToString()     ?? "Level 0";
                string topLevelName = req["top_level"]?.ToString();
                var extraParams     = req["parameters"] as JObject;

                var result = (JObject)RunOnRevitThread(uiapp =>
                {
                    var doc = state.Document;

                    // Find symbol
                    var symbol = new FilteredElementCollector(doc)
                        .OfClass(typeof(FamilySymbol))
                        .Cast<FamilySymbol>()
                        .FirstOrDefault(s =>
                            string.Equals(s.Family.Name, famName, StringComparison.OrdinalIgnoreCase) &&
                            string.Equals(s.Name, typeName, StringComparison.OrdinalIgnoreCase));

                    if (symbol == null)
                        throw new Exception(
                            $"Family type '{famName}::{typeName}' not found in document. " +
                            "Call /load-family first.");

                    // Find base level
                    Level baseLevel = GetLevel(doc, levelName);
                    if (baseLevel == null)
                        throw new Exception($"Level '{levelName}' not found in document.");

                    XYZ location = new XYZ(xMm / 304.8, yMm / 304.8, zMm / 304.8);
                    int  catInt  = symbol.Category?.Id?.IntegerValue ?? 0;

                    var wc = new WarningCollector(@"C:\RevitOutput\session_log.txt");
                    FamilyInstance inst = null;

                    using (var t = new Transaction(doc, $"Place {famName}::{typeName}"))
                    {
                        var fo = t.GetFailureHandlingOptions();
                        fo.SetFailuresPreprocessor(wc);
                        t.SetFailureHandlingOptions(fo);
                        t.Start();

                        // Activate symbol (must be inside a transaction)
                        if (!symbol.IsActive) symbol.Activate();
                        doc.Regenerate();

                        // Choose placement overload based on category
                        if (catInt == (int)BuiltInCategory.OST_StructuralColumns)
                            inst = doc.Create.NewFamilyInstance(
                                       location, symbol, baseLevel,
                                       StructuralType.Column);
                        else if (catInt == (int)BuiltInCategory.OST_StructuralFraming)
                            inst = doc.Create.NewFamilyInstance(
                                       location, symbol, baseLevel,
                                       StructuralType.Beam);
                        else
                            inst = doc.Create.NewFamilyInstance(
                                       location, symbol, baseLevel,
                                       StructuralType.NonStructural);

                        // Top-level constraint (structural columns)
                        if (catInt == (int)BuiltInCategory.OST_StructuralColumns
                            && !string.IsNullOrEmpty(topLevelName))
                        {
                            Level topLevel = GetLevel(doc, topLevelName);
                            if (topLevel != null && topLevel.Elevation > baseLevel.Elevation + 1e-6)
                                inst.get_Parameter(BuiltInParameter.FAMILY_TOP_LEVEL_PARAM)
                                    ?.Set(topLevel.Id);
                            else if (topLevel == null)
                            {
                                // Fall back to first level above base
                                Level fallback = GetLevelAbove(doc, baseLevel);
                                if (fallback != null)
                                    inst.get_Parameter(BuiltInParameter.FAMILY_TOP_LEVEL_PARAM)
                                        ?.Set(fallback.Id);
                            }
                        }

                        // Extra instance parameters
                        if (extraParams != null)
                        {
                            foreach (var kv in extraParams)
                                SetInstanceParam(inst, symbol, kv.Key, kv.Value);
                        }

                        t.Commit();
                    }

                    wc.LogSummary($"Session {sessionId} place {famName}::{typeName}");

                    if (inst == null)
                        throw new Exception("NewFamilyInstance returned null — check the family type and level.");

                    // Record placed element in session
                    var pe = new SessionElement {
                        ElementId  = inst.Id.IntegerValue.ToString(),
                        Category   = symbol.Category?.Name ?? "Unknown",
                        FamilyName = famName,
                        TypeName   = typeName,
                        X_mm = xMm, Y_mm = yMm, Z_mm = zMm,
                        Level = levelName,
                        Warnings = wc.Warnings,
                    };
                    state.AddElement(pe);

                    Log($"Session {sessionId}: placed {famName}::{typeName} at " +
                        $"({xMm:F0},{yMm:F0}) level={levelName}  elemId={inst.Id.IntegerValue}" +
                        (wc.Warnings.Count > 0 ? $"  [{wc.Warnings.Count} warning(s)]" : ""));

                    return new JObject {
                        ["element_id"] = inst.Id.IntegerValue.ToString(),
                        ["placed"]     = JObject.FromObject(pe),
                        ["warnings"]   = new JArray(wc.Warnings.Select(w => (JToken)w)),
                    };
                });

                WriteJson(stream, 200, result);
            }
            catch (Exception ex)
            {
                WriteJson(stream, 500, new { error = ex.Message });
            }
        }

        // Sets a parameter on a FamilyInstance from a JToken value.
        // Numeric values are assumed to be in mm → converted to internal feet.
        private static void SetInstanceParam(FamilyInstance inst, FamilySymbol sym,
                                             string name, JToken value)
        {
            var param = inst.LookupParameter(name) ?? sym.LookupParameter(name);
            if (param == null || param.IsReadOnly) return;

            switch (param.StorageType)
            {
                case StorageType.Double:
                    double raw = value.Value<double>();
                    // Heuristic: if the name suggests a length-type param, convert mm→ft
                    param.Set(raw / 304.8);
                    break;
                case StorageType.Integer:
                    param.Set(value.Value<int>());
                    break;
                case StorageType.String:
                    param.Set(value.ToString());
                    break;
                case StorageType.ElementId:
                    param.Set(new ElementId(value.Value<int>()));
                    break;
            }
        }

        // =========================================================================
        // Handler — POST /session/{id}/set-param
        //
        // Body: {
        //   "element_id":     "12345",
        //   "parameter_name": "Mark",
        //   "value":          "C1",
        //   "value_type":     "string"   // "mm" | "raw" | "string" | "int" | "id"
        // }
        // =========================================================================

        private void HandleSetParam(NetworkStream stream, string sessionId, string body)
        {
            var state = _sessions.Get(sessionId);
            if (state == null) { WriteJson(stream, 404, new { error = "Session not found" }); return; }

            try
            {
                var req       = JObject.Parse(body);
                string elemId = req["element_id"]?.ToString()     ?? throw new Exception("element_id required");
                string pName  = req["parameter_name"]?.ToString() ?? throw new Exception("parameter_name required");
                var    value  = req["value"]                      ?? throw new Exception("value required");
                string vType  = req["value_type"]?.ToString()     ?? "mm";

                var result = (JObject)RunOnRevitThread(uiapp =>
                {
                    var doc  = state.Document;
                    var elem = doc.GetElement(new ElementId(int.Parse(elemId)));
                    if (elem == null) throw new Exception($"Element {elemId} not found in session.");

                    var param = elem.LookupParameter(pName);
                    if (param == null) throw new Exception($"Parameter '{pName}' not found on element.");
                    if (param.IsReadOnly) throw new Exception($"Parameter '{pName}' is read-only.");

                    using (var t = new Transaction(doc, $"Set {pName}"))
                    {
                        t.Start();
                        switch (vType)
                        {
                            case "mm":     param.Set(value.Value<double>() / 304.8); break;
                            case "raw":    param.Set(value.Value<double>());          break;
                            case "string": param.Set(value.ToString());               break;
                            case "int":    param.Set(value.Value<int>());             break;
                            case "id":     param.Set(new ElementId(value.Value<int>())); break;
                            default:       throw new Exception($"Unknown value_type: {vType}");
                        }
                        t.Commit();
                    }

                    string displayVal = vType == "mm"
                        ? $"{value.Value<double>()} mm"
                        : value.ToString();
                    Log($"Session {sessionId}: set param '{pName}'={displayVal} on elem {elemId}");

                    return new JObject {
                        ["ok"]             = true,
                        ["element_id"]     = elemId,
                        ["parameter_name"] = pName,
                        ["value_set"]      = displayVal,
                    };
                });

                WriteJson(stream, 200, result);
            }
            catch (Exception ex)
            {
                WriteJson(stream, 500, new { error = ex.Message });
            }
        }

        // =========================================================================
        // Handler — POST /session/{id}/get-params
        // Body: { "element_id": "12345" }
        // Returns all parameters of the element so the agent can discover
        // the correct parameter names before calling set-param.
        // =========================================================================

        private void HandleGetParams(NetworkStream stream, string sessionId, string body)
        {
            var state = _sessions.Get(sessionId);
            if (state == null) { WriteJson(stream, 404, new { error = "Session not found" }); return; }

            try
            {
                var req = JObject.Parse(body);
                string eid = req["element_id"]?.ToString() ?? throw new Exception("element_id required");

                var result = (JObject)RunOnRevitThread(uiapp =>
                {
                    var doc  = state.Document;
                    var elem = doc.GetElement(new ElementId(int.Parse(eid)));
                    if (elem == null)
                        throw new Exception($"Element {eid} not found in session.");

                    var arr = new JArray();
                    foreach (Parameter p in elem.Parameters)
                    {
                        try
                        {
                            string defName = p.Definition?.Name;
                            if (string.IsNullOrWhiteSpace(defName)) continue;

                            var obj = new JObject
                            {
                                ["name"]         = defName,
                                ["storage_type"] = p.StorageType.ToString(),
                                ["is_read_only"] = p.IsReadOnly,
                            };
                            switch (p.StorageType)
                            {
                                case StorageType.Double:
                                    obj["value_mm"] = Math.Round(p.AsDouble() * 304.8, 2);
                                    break;
                                case StorageType.Integer:
                                    obj["value_int"] = p.AsInteger();
                                    break;
                                case StorageType.String:
                                    obj["value_str"] = p.AsString() ?? "";
                                    break;
                                case StorageType.ElementId:
                                    obj["value_id"] = p.AsElementId().IntegerValue;
                                    break;
                            }
                            arr.Add(obj);
                        }
                        catch { }
                    }

                    // Sort: editable first, then alphabetically
                    var sorted = arr
                        .OrderBy(o => (bool)((JObject)o)["is_read_only"])
                        .ThenBy(o => (string)((JObject)o)["name"])
                        .ToList();

                    Log($"Session {sessionId}: get-params elem {eid} → {sorted.Count} parameter(s)");
                    return new JObject
                    {
                        ["element_id"] = eid,
                        ["count"]      = sorted.Count,
                        ["parameters"] = new JArray(sorted),
                    };
                });

                WriteJson(stream, 200, result);
            }
            catch (Exception ex)
            {
                WriteJson(stream, 500, new { error = ex.Message });
            }
        }

        // =========================================================================
        // Handler — POST /session/{id}/wall-join
        // Enables wall joins at both ends of every Wall in the session.
        // Fixes display gaps and incorrect intersections at T-junctions / corners.
        // =========================================================================

        private void HandleWallJoinAll(NetworkStream stream, string sessionId)
        {
            var state = _sessions.Get(sessionId);
            if (state == null) { WriteJson(stream, 404, new { error = "Session not found" }); return; }

            try
            {
                var result = (JObject)RunOnRevitThread(uiapp =>
                {
                    var doc   = state.Document;
                    var walls = new FilteredElementCollector(doc)
                                    .OfClass(typeof(Wall))
                                    .Cast<Wall>()
                                    .ToList();

                    int joined = 0;
                    var wc = new WarningCollector(@"C:\RevitOutput\session_log.txt");
                    using (var t = new Transaction(doc, "Join All Walls"))
                    {
                        var fo = t.GetFailureHandlingOptions();
                        fo.SetFailuresPreprocessor(wc);
                        t.SetFailureHandlingOptions(fo);
                        t.Start();
                        foreach (var wall in walls)
                        {
                            try
                            {
                                WallUtils.AllowWallJoinAtEnd(wall, 0);
                                WallUtils.AllowWallJoinAtEnd(wall, 1);
                                joined++;
                            }
                            catch { }
                        }
                        t.Commit();
                    }

                    wc.LogSummary($"Session {sessionId} wall-join-all");
                    Log($"Session {sessionId}: wall-join-all — {joined}/{walls.Count} walls processed" +
                        (wc.Warnings.Count > 0 ? $"  [{wc.Warnings.Count} warning(s)]" : "") + ".");
                    return new JObject
                    {
                        ["ok"]           = true,
                        ["walls_total"]  = walls.Count,
                        ["walls_joined"] = joined,
                        ["warnings"]     = new JArray(wc.Warnings.Select(w => (JToken)w)),
                    };
                });

                WriteJson(stream, 200, result);
            }
            catch (Exception ex)
            {
                WriteJson(stream, 500, new { error = ex.Message });
            }
        }

        // =========================================================================
        // Handler — POST /session/{id}/export-view
        // Exports the first floor plan view as a PNG (150 DPI, 2048 px wide).
        // Used by the Python vision comparator to verify model accuracy after build.
        // Returns binary PNG bytes with Content-Type: image/png.
        // =========================================================================

        private void HandleExportView(NetworkStream stream, string sessionId)
        {
            var state = _sessions.Get(sessionId);
            if (state == null) { WriteJson(stream, 404, new { error = "Session not found" }); return; }

            try
            {
                string pngPath = null;
                RunOnRevitThread(uiapp =>
                {
                    var doc = state.Document;

                    // Prefer a named "Ground Floor Plan"; fall back to first non-template floor plan
                    ViewPlan fp = new FilteredElementCollector(doc)
                        .OfClass(typeof(ViewPlan))
                        .Cast<ViewPlan>()
                        .Where(v => !v.IsTemplate && v.ViewType == ViewType.FloorPlan)
                        .OrderBy(v => v.Name.Contains("Ground") ? 0 : 1)
                        .FirstOrDefault();

                    if (fp == null)
                        throw new Exception("No floor plan view found in session document.");

                    string outputDir = @"C:\RevitOutput";
                    string baseName  = $"view_{sessionId}";
                    string basePath  = Path.Combine(outputDir, baseName);

                    var opts = new ImageExportOptions
                    {
                        ExportRange           = ExportRange.SetOfViews,
                        ImageResolution       = ImageResolution.DPI_150,
                        FilePath              = basePath,
                        HLRandWFViewsFileType = ImageFileType.PNG,
                        ShadowViewsFileType   = ImageFileType.PNG,
                        ZoomType              = ZoomFitType.FitToPage,
                        PixelSize             = 2048,
                    };
                    opts.SetViewsAndSheets(new List<ElementId> { fp.Id });
                    doc.ExportImage(opts);

                    // Revit appends view type + name: basePath - Floor Plan - <name>.png
                    var candidates = Directory.GetFiles(
                        outputDir, $"{baseName}*.png", SearchOption.TopDirectoryOnly);
                    if (candidates.Length == 0)
                        throw new Exception("ExportImage produced no PNG output file.");

                    // Pick most recently written match
                    pngPath = candidates.OrderByDescending(File.GetLastWriteTime).First();
                    Log($"Session {sessionId}: export-view → {pngPath}");
                    return null;
                });

                byte[] pngBytes = File.ReadAllBytes(pngPath);
                string respHeader =
                    $"HTTP/1.1 200 OK\r\n" +
                    $"Content-Type: image/png\r\n" +
                    $"Content-Length: {pngBytes.Length}\r\n" +
                    $"Content-Disposition: inline; filename=floorplan_{sessionId}.png\r\n" +
                    $"Connection: close\r\n\r\n";

                byte[] hdrBytes = Encoding.UTF8.GetBytes(respHeader);
                stream.Write(hdrBytes, 0, hdrBytes.Length);
                stream.Write(pngBytes, 0, pngBytes.Length);
                Console.WriteLine($"[RevitModelBuilderAddin] View export {sessionId}: {pngBytes.Length:N0} bytes");
            }
            catch (Exception ex)
            {
                WriteJson(stream, 500, new { error = ex.Message });
            }
        }

        // =========================================================================
        // Handler — GET /session/{id}/state
        // =========================================================================

        private void HandleQueryState(NetworkStream stream, string sessionId)
        {
            var state = _sessions.Get(sessionId);
            if (state == null) { WriteJson(stream, 404, new { error = "Session not found" }); return; }

            try
            {
                var result = (JObject)RunOnRevitThread(uiapp =>
                {
                    var doc = state.Document;

                    var levels = new FilteredElementCollector(doc)
                        .OfClass(typeof(Level)).Cast<Level>()
                        .OrderBy(l => l.Elevation)
                        .Select(l => new JObject {
                            ["name"]         = l.Name,
                            ["elevation_mm"] = Math.Round(l.Elevation * 304.8, 1),
                        }).ToArray();

                    var families = new FilteredElementCollector(doc)
                        .OfClass(typeof(Family)).Cast<Family>()
                        .OrderBy(f => f.Name)
                        .Select(f => new JObject {
                            ["name"]       = f.Name,
                            ["type_count"] = f.GetFamilySymbolIds().Count,
                        }).ToArray();

                    var placed = state.GetElements().Select(e => JObject.FromObject(e)).ToArray();

                    return new JObject {
                        ["session_id"]      = sessionId,
                        ["levels"]          = new JArray(levels),
                        ["loaded_families"] = new JArray(families),
                        ["placed_elements"] = new JArray(placed),
                        ["placed_count"]    = placed.Length,
                    };
                });

                WriteJson(stream, 200, result);
            }
            catch (Exception ex)
            {
                WriteJson(stream, 500, new { error = ex.Message });
            }
        }

        // =========================================================================
        // Handler — POST /session/{id}/query-elements
        // Body: { "element_ids": [int]?, "category": string? }
        //       element_ids  — specific Revit IDs to fetch (takes precedence over category)
        //       category     — BuiltInCategory short name ("StructuralColumns") or
        //                      full name ("OST_StructuralColumns"); filters a collector pass
        //       both omitted — every element with a valid BoundingBox in the session doc
        //
        // Used by the self-healing agent's clash resolver to obtain current
        // bounding boxes after placement (feet → mm at the C# boundary).
        // =========================================================================

        private void HandleQueryElements(NetworkStream stream, string sessionId, string body)
        {
            var state = _sessions.Get(sessionId);
            if (state == null) { WriteJson(stream, 404, new { error = "Session not found" }); return; }

            List<int> requestedIds = null;
            BuiltInCategory? categoryFilter = null;

            try
            {
                if (!string.IsNullOrWhiteSpace(body))
                {
                    var req = JObject.Parse(body);

                    var idsTok = req["element_ids"] as JArray;
                    if (idsTok != null && idsTok.Count > 0)
                        requestedIds = idsTok.Select(t => t.Value<int>()).ToList();

                    var catTok = req["category"]?.ToString();
                    if (!string.IsNullOrWhiteSpace(catTok))
                    {
                        string key = catTok.StartsWith("OST_") ? catTok : "OST_" + catTok;
                        if (Enum.TryParse(key, out BuiltInCategory parsed))
                            categoryFilter = parsed;
                        else
                            throw new Exception($"Unknown category '{catTok}'.");
                    }
                }
            }
            catch (Exception ex)
            {
                WriteJson(stream, 400, new { error = $"Bad request: {ex.Message}" });
                return;
            }

            try
            {
                var result = (JObject)RunOnRevitThread(uiapp =>
                {
                    var doc = state.Document;
                    IEnumerable<Element> elems;
                    bool unfilteredScan = requestedIds == null;

                    if (requestedIds != null)
                    {
                        elems = requestedIds
                            .Select(i => doc.GetElement(new ElementId(i)))
                            .Where(e => e != null);
                    }
                    else
                    {
                        var fec = new FilteredElementCollector(doc).WhereElementIsNotElementType();
                        if (categoryFilter.HasValue) fec = fec.OfCategory(categoryFilter.Value);
                        elems = fec;   // capped below by scan count, not materialized early
                    }

                    // Cache ElementType lookups: many elements share a type, so
                    // doc.GetElement(typeId) is redundant for all but the first.
                    var typeCache = new Dictionary<ElementId, ElementType>();
                    ElementType LookupType(ElementId tid)
                    {
                        if (tid == null || tid == ElementId.InvalidElementId) return null;
                        if (!typeCache.TryGetValue(tid, out var t))
                        {
                            t = doc.GetElement(tid) as ElementType;
                            typeCache[tid] = t;
                        }
                        return t;
                    }

                    JObject Mm(XYZ p) => new JObject
                    {
                        ["x"] = Math.Round(p.X * FEET_TO_MM, 1),
                        ["y"] = Math.Round(p.Y * FEET_TO_MM, 1),
                        ["z"] = Math.Round(p.Z * FEET_TO_MM, 1),
                    };

                    var arr = new JArray();
                    bool truncated = false;
                    int  scanned   = 0;
                    foreach (var elem in elems)
                    {
                        // Cap by scan count (not arr.Count) so skipped non-geometric
                        // elements still consume budget — otherwise a model full of
                        // detail lines would never hit the ceiling.
                        if (unfilteredScan && scanned >= QUERY_ELEMENTS_MAX_SCAN)
                        {
                            truncated = true;
                            break;
                        }
                        scanned++;

                        BoundingBoxXYZ bb;
                        try { bb = elem.get_BoundingBox(null); } catch { bb = null; }
                        if (bb == null) continue;       // skip non-geometric elements

                        var sym = LookupType(elem.GetTypeId());

                        var obj = new JObject
                        {
                            ["id"]       = elem.Id.IntegerValue,
                            ["category"] = elem.Category?.Name ?? "",
                            ["family"]   = sym?.FamilyName ?? "",
                            ["type"]     = sym?.Name       ?? "",
                            ["bbox"]     = new JObject { ["min"] = Mm(bb.Min), ["max"] = Mm(bb.Max) },
                        };

                        if (elem.Location is LocationPoint lp)
                            obj["location"] = Mm(lp.Point);

                        arr.Add(obj);
                    }

                    Log($"Session {sessionId}: query-elements → {arr.Count} element(s)"
                        + (truncated ? " (truncated)" : ""));
                    var response = new JObject { ["elements"] = arr };
                    if (truncated) response["truncated"] = true;
                    return response;
                });

                WriteJson(stream, 200, result);
            }
            catch (Exception ex)
            {
                WriteJson(stream, 500, new { error = ex.Message });
            }
        }

        // =========================================================================
        // Handler — POST /session/{id}/export
        // Body: { "keep_open": false }
        // =========================================================================

        private void HandleExportSession(NetworkStream stream, string sessionId, string body)
        {
            var state = _sessions.Get(sessionId);
            if (state == null) { WriteJson(stream, 404, new { error = "Session not found" }); return; }

            bool keepOpen = false;
            try
            {
                if (!string.IsNullOrWhiteSpace(body))
                    keepOpen = JObject.Parse(body)["keep_open"]?.Value<bool>() ?? false;
            }
            catch { }

            try
            {
                string outputDir  = @"C:\RevitOutput";
                Directory.CreateDirectory(outputDir);
                string outputPath = Path.Combine(outputDir, $"session_{sessionId}.rvt");

                RunOnRevitThread(uiapp =>
                {
                    var doc = state.Document;

                    // Create views in a separate transaction
                    try
                    {
                        using (var vt = new Transaction(doc, "Create Views"))
                        {
                            vt.Start();
                            CreateSessionViews(doc);
                            vt.Commit();
                        }
                    }
                    catch (Exception vex)
                    {
                        Log($"Session {sessionId}: view creation skipped — {vex.Message}");
                    }

                    var saveOpts = new SaveAsOptions { OverwriteExistingFile = true };
                    doc.SaveAs(outputPath, saveOpts);
                    Log($"Session {sessionId}: exported to {outputPath}");
                    return null;
                });

                byte[] rvtBytes = File.ReadAllBytes(outputPath);

                if (!keepOpen)
                {
                    RunOnRevitThread(uiapp =>
                    {
                        try { state.Document.Close(false); } catch { }
                        return null;
                    });
                    _sessions.Remove(sessionId);
                    Log($"Session {sessionId}: closed after export.");
                }

                // Collect families for the manifest header
                string familiesJson = "{}";
                if (_buildHandler?.ResultFamilies != null)
                    familiesJson = JsonConvert.SerializeObject(_buildHandler.ResultFamilies);

                string respHeader =
                    $"HTTP/1.1 200 OK\r\n" +
                    $"Content-Type: application/octet-stream\r\n" +
                    $"Content-Length: {rvtBytes.Length}\r\n" +
                    $"Content-Disposition: attachment; filename=session_{sessionId}.rvt\r\n" +
                    $"X-Session-Id: {sessionId}\r\n" +
                    $"X-Session-Closed: {(!keepOpen).ToString().ToLower()}\r\n" +
                    $"Connection: close\r\n\r\n";

                byte[] hdrBytes = Encoding.UTF8.GetBytes(respHeader);
                stream.Write(hdrBytes, 0, hdrBytes.Length);
                stream.Write(rvtBytes, 0, rvtBytes.Length);
                Console.WriteLine($"[RevitModelBuilderAddin] Session {sessionId} exported ({rvtBytes.Length:N0} bytes)");
            }
            catch (Exception ex)
            {
                WriteJson(stream, 500, new { error = ex.Message });
            }
        }

        // Create minimal floor plan + 3D views for session exports
        private static void CreateSessionViews(Document doc)
        {
            Level level0 = GetLevel(doc, "Level 0")
                        ?? new FilteredElementCollector(doc)
                               .OfClass(typeof(Level)).Cast<Level>()
                               .OrderBy(l => l.Elevation).FirstOrDefault();
            if (level0 == null) return;

            var fpVFT = new FilteredElementCollector(doc)
                .OfClass(typeof(ViewFamilyType)).Cast<ViewFamilyType>()
                .FirstOrDefault(v => v.ViewFamily == ViewFamily.FloorPlan);
            if (fpVFT != null)
            {
                var vp = ViewPlan.Create(doc, fpVFT.Id, level0.Id);
                vp.Name = "Ground Floor Plan";
            }

            var tdVFT = new FilteredElementCollector(doc)
                .OfClass(typeof(ViewFamilyType)).Cast<ViewFamilyType>()
                .FirstOrDefault(v => v.ViewFamily == ViewFamily.ThreeDimensional);
            if (tdVFT != null)
            {
                var v3d = View3D.CreateIsometric(doc, tdVFT.Id);
                v3d.Name = "3D Overview";
            }
        }

        // =========================================================================
        // Handler — POST /session/{id}/close
        // =========================================================================

        private void HandleCloseSession(NetworkStream stream, string sessionId)
        {
            var state = _sessions.Get(sessionId);
            if (state == null) { WriteJson(stream, 404, new { error = "Session not found" }); return; }

            try
            {
                RunOnRevitThread(uiapp =>
                {
                    try { state.Document.Close(false); } catch { }
                    return null;
                });
                _sessions.Remove(sessionId);
                Log($"Session {sessionId}: closed.");
                WriteJson(stream, 200, new { ok = true, message = $"Session {sessionId} closed." });
            }
            catch (Exception ex)
            {
                WriteJson(stream, 500, new { error = ex.Message });
            }
        }

        // =========================================================================
        // Shared geometry helpers (referenced by both handlers)
        // =========================================================================

        private static Level GetLevel(Document doc, string name) =>
            new FilteredElementCollector(doc)
                .OfClass(typeof(Level)).Cast<Level>()
                .FirstOrDefault(l =>
                    string.Equals(l.Name, name, StringComparison.OrdinalIgnoreCase));

        private static Level GetLevelAbove(Document doc, Level baseLevel) =>
            new FilteredElementCollector(doc)
                .OfClass(typeof(Level)).Cast<Level>()
                .Where(l => l.Elevation > baseLevel.Elevation + 1e-6)
                .OrderBy(l => l.Elevation)
                .FirstOrDefault();

        // =========================================================================
        // TCP helpers
        // =========================================================================

        private static string ReadLine(NetworkStream stream)
        {
            var sb = new StringBuilder();
            int b;
            while ((b = stream.ReadByte()) != -1)
            {
                if (b == '\r') { stream.ReadByte(); break; }
                if (b == '\n') break;
                sb.Append((char)b);
            }
            return sb.ToString();
        }

        private static void WriteText(NetworkStream stream, int code, string text)
        {
            byte[] body   = Encoding.UTF8.GetBytes(text);
            string status = code == 200 ? "OK" : code == 404 ? "Not Found" : "Error";
            string resp   =
                $"HTTP/1.1 {code} {status}\r\n" +
                $"Content-Type: text/plain\r\n" +
                $"Content-Length: {body.Length}\r\n" +
                $"Connection: close\r\n\r\n";
            byte[] respBytes = Encoding.UTF8.GetBytes(resp);
            stream.Write(respBytes, 0, respBytes.Length);
            stream.Write(body, 0, body.Length);
        }

        private static void WriteJson(NetworkStream stream, int code, object payload)
        {
            string json   = payload is JToken jt
                ? jt.ToString(Formatting.None)
                : JsonConvert.SerializeObject(payload);
            byte[] body   = Encoding.UTF8.GetBytes(json);
            string status = code == 200 ? "OK"
                          : code == 400 ? "Bad Request"
                          : code == 404 ? "Not Found"
                          : "Internal Server Error";
            string resp   =
                $"HTTP/1.1 {code} {status}\r\n" +
                $"Content-Type: application/json\r\n" +
                $"Content-Length: {body.Length}\r\n" +
                $"Connection: close\r\n\r\n";
            byte[] respBytes = Encoding.UTF8.GetBytes(resp);
            stream.Write(respBytes, 0, respBytes.Length);
            stream.Write(body, 0, body.Length);
        }

        private static void Log(string msg)
        {
            const string LOG = @"C:\RevitOutput\session_log.txt";
            try { File.AppendAllText(LOG, $"[{DateTime.Now:HH:mm:ss}] {msg}\r\n"); } catch { }
        }

        /// <summary>
        /// Build the .rvt filename: "{safe_pdf_stem}_{jobId}.rvt" or "{jobId}.rvt".
        /// </summary>
        private static string RvtFileName(string pdfFilename, string jobId)
        {
            if (!string.IsNullOrWhiteSpace(pdfFilename))
            {
                string stem = System.IO.Path.GetFileNameWithoutExtension(pdfFilename);
                string safe = System.Text.RegularExpressions.Regex
                    .Replace(stem, @"[^\w\-]", "_");
                safe = System.Text.RegularExpressions.Regex
                    .Replace(safe, @"_+", "_")
                    .Trim('_');
                if (!string.IsNullOrEmpty(safe))
                    return $"{safe}_{jobId}.rvt";
            }
            return $"{jobId}.rvt";
        }
    }

    // =========================================================================
    // CommandHandler — single-step IExternalEventHandler
    // Uses a Func<UIApplication, object> delegate set by the HTTP thread before
    // the event is raised, executed on the Revit UI thread, result returned.
    // =========================================================================

    public class CommandHandler : IExternalEventHandler
    {
        private Func<UIApplication, object> _command;
        private object                      _result;
        private Exception                   _error;
        private readonly ManualResetEventSlim _done = new ManualResetEventSlim(false);

        /// <summary>Call from HTTP thread before raising the ExternalEvent.</summary>
        public void SetCommand(Func<UIApplication, object> work)
        {
            _command = work;
            _result  = null;
            _error   = null;
            _done.Reset();
        }

        /// <summary>Called by Revit UI thread.</summary>
        public void Execute(UIApplication app)
        {
            try   { _result = _command?.Invoke(app); }
            catch (Exception ex) { _error = ex; }
            finally { _done.Set(); }
        }

        /// <summary>Block HTTP thread until result is ready.</summary>
        public object WaitResult(int timeoutMs = 60_000)
        {
            if (!_done.Wait(timeoutMs))
                throw new TimeoutException($"Revit command timed out after {timeoutMs / 1000} s.");
            if (_error != null)
                System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(_error).Throw();
            return _result;
        }

        public string GetName() => "RevitModelBuilderAddin.CommandHandler";
    }

    // =========================================================================
    // RevitSessionManager — holds open Document references between HTTP calls
    // =========================================================================

    public class RevitSessionManager
    {
        private readonly ConcurrentDictionary<string, SessionState> _sessions
            = new ConcurrentDictionary<string, SessionState>(StringComparer.Ordinal);

        public string Create(Document doc, string templatePath)
        {
            string id = Guid.NewGuid().ToString("N").Substring(0, 12);
            _sessions[id] = new SessionState(doc, templatePath);
            return id;
        }

        public SessionState Get(string id) =>
            _sessions.TryGetValue(id, out var s) ? s : null;

        public void Remove(string id) => _sessions.TryRemove(id, out _);

        public void CloseAll()
        {
            foreach (var kv in _sessions)
                try { kv.Value.Document.Close(false); } catch { }
            _sessions.Clear();
        }
    }

    public class SessionState
    {
        private readonly List<SessionElement> _elements = new List<SessionElement>();
        private readonly object _lock = new object();

        public Document  Document     { get; }
        public string    TemplatePath { get; }
        public DateTime  CreatedAt    { get; } = DateTime.UtcNow;

        public SessionState(Document doc, string templatePath)
        {
            Document     = doc;
            TemplatePath = templatePath;
        }

        public void AddElement(SessionElement e)
        {
            lock (_lock) { _elements.Add(e); }
        }

        public IReadOnlyList<SessionElement> GetElements()
        {
            lock (_lock) { return _elements.ToList(); }
        }
    }

    public class SessionElement
    {
        public string       ElementId  { get; set; }
        public string       Category   { get; set; }
        public string       FamilyName { get; set; }
        public string       TypeName   { get; set; }
        public double       X_mm       { get; set; }
        public double       Y_mm       { get; set; }
        public double       Z_mm       { get; set; }
        public string       Level      { get; set; }
        public List<string> Warnings   { get; set; } = new List<string>();
    }

    // =========================================================================
    // IFamilyLoadOptions — always overwrite (used for session load-family calls)
    // =========================================================================

    public class OverwriteFamilyLoadOptions : IFamilyLoadOptions
    {
        public bool OnFamilyFound(bool familyInUse, out bool overwriteParameterValues)
        {
            overwriteParameterValues = true;
            return true;
        }

        public bool OnSharedFamilyFound(Family sharedFamily, bool familyInUse,
                                        out FamilySource source, out bool overwriteParameterValues)
        {
            source = FamilySource.Family;
            overwriteParameterValues = true;
            return true;
        }
    }

    // =========================================================================
    // FailureHelpers — shared logic for Revit failure-message resolution.
    // Used by both the per-transaction WarningCollector and the application-
    // level OnFailuresProcessing safety net.
    // =========================================================================

    internal static class FailureHelpers
    {
        // Preference order for "Cannot keep elements joined" resolutions.
        // UnjoinElements matches the "Unjoin Elements" button on Revit's native
        // dialog and is the canonical fix; SkipElements/DetachElements are
        // fallbacks for join failures that don't expose UnjoinElements.
        public static readonly FailureResolutionType[] JoinResolutionLadder =
        {
            FailureResolutionType.UnjoinElements,
            FailureResolutionType.SkipElements,
            FailureResolutionType.DetachElements,
        };

        public static bool IsJoinError(string text)
        {
            if (string.IsNullOrEmpty(text)) return false;
            return text.IndexOf("cannot keep", StringComparison.OrdinalIgnoreCase) >= 0
                || (text.IndexOf("element",    StringComparison.OrdinalIgnoreCase) >= 0
                 && text.IndexOf("join",       StringComparison.OrdinalIgnoreCase) >= 0);
        }

        // Try JoinResolutionLadder in order; returns the type applied, or null
        // if nothing resolved.
        public static FailureResolutionType? TryApplyJoinResolution(
            FailuresAccessor fa, FailureMessageAccessor msg)
        {
            foreach (var resType in JoinResolutionLadder)
            {
                if (!msg.HasResolutionOfType(resType)) continue;
                try
                {
                    msg.SetCurrentResolutionType(resType);
                    fa.ResolveFailure(msg);
                    return resType;
                }
                catch { /* try next resolution type */ }
            }
            return null;
        }
    }

    // =========================================================================
    // WarningCollector — IFailuresPreprocessor (used by both handlers)
    // =========================================================================

    public class WarningCollector : IFailuresPreprocessor
    {
        // Optional path — when provided every warning/error is written here
        // immediately as it is intercepted (before the transaction commits).
        private readonly string _logPath;

        public List<string> Warnings { get; } = new List<string>();
        public List<string> Errors   { get; } = new List<string>();

        // Detailed records: message text + element IDs of affected elements.
        public List<(string Text, List<int> ElementIds)> WarningDetails { get; }
            = new List<(string, List<int>)>();
        public List<(string Text, List<int> ElementIds)> ErrorDetails { get; }
            = new List<(string, List<int>)>();

        public WarningCollector(string logPath = null) { _logPath = logPath; }

        public FailureProcessingResult PreprocessFailures(FailuresAccessor fa)
        {
            foreach (var msg in fa.GetFailureMessages().ToList())
            {
                string text    = msg.GetDescriptionText();
                var    elemIds = msg.GetFailingElementIds()
                                    .Select(id => id.IntegerValue).ToList();
                string elemStr = elemIds.Count > 0
                    ? $" — affected element IDs: [{string.Join(", ", elemIds)}]"
                    : "";

                if (msg.GetSeverity() == FailureSeverity.Warning)
                {
                    Warnings.Add(text);
                    WarningDetails.Add((text, elemIds));
                    fa.DeleteWarning(msg);
                    WriteLog($"[WARN] {text}{elemStr}");
                }
                else
                {
                    // "Cannot keep elements joined" — Revit tries to auto-join
                    // adjacent structural members (column↔beam) and fails when
                    // geometry is complex.  Resolution accepts the unjoin
                    // (elements remain; join is removed) — equivalent to
                    // clicking "Unjoin" in Revit's error dialog.
                    FailureResolutionType? applied = FailureHelpers.IsJoinError(text)
                        ? FailureHelpers.TryApplyJoinResolution(fa, msg)
                        : null;

                    if (applied != null)
                    {
                        string tagged = $"[JOIN-RESOLVED:{applied}] {text}";
                        Warnings.Add(tagged);
                        WarningDetails.Add((tagged, elemIds));
                        WriteLog($"{tagged}{elemStr}");
                    }
                    else
                    {
                        Errors.Add(text);
                        ErrorDetails.Add((text, elemIds));
                        WriteLog($"[ERROR] severity={msg.GetSeverity()} {text}{elemStr}");
                    }
                }
            }
            return FailureProcessingResult.Continue;
        }

        /// <summary>
        /// Write a one-line summary after a transaction completes.
        /// Call this right after t.Commit() wherever a WarningCollector is used.
        /// </summary>
        public void LogSummary(string context = "")
        {
            if (Warnings.Count == 0 && Errors.Count == 0) return;
            string prefix = string.IsNullOrEmpty(context) ? "" : $"[{context}] ";
            if (Warnings.Count > 0)
                WriteLog($"{prefix}{Warnings.Count} warning(s): " +
                         string.Join(" | ", Warnings));
            if (Errors.Count > 0)
                WriteLog($"{prefix}{Errors.Count} error(s): " +
                         string.Join(" | ", Errors));
        }

        private void WriteLog(string message)
        {
            if (_logPath == null) return;
            try { File.AppendAllText(_logPath,
                      $"[{DateTime.Now:HH:mm:ss}] {message}\r\n"); }
            catch { }
        }
    }

    // =========================================================================
    // BuildHandler — monolithic batch IExternalEventHandler (unchanged from v1)
    // =========================================================================

    public class BuildHandler : IExternalEventHandler
    {
        private string _transactionJson;
        private string _outputPath;

        public ManualResetEventSlim Done       { get; } = new ManualResetEventSlim(false);
        public Exception            BuildError { get; private set; }
        public List<(string Text, List<int> ElementIds)> ResultWarningDetails { get; private set; }
            = new List<(string, List<int>)>();
        public Dictionary<string, List<string>> ResultFamilies { get; private set; }
            = new Dictionary<string, List<string>>();

        public void Prepare(string transactionJson, string outputPath)
        {
            _transactionJson     = transactionJson;
            _outputPath          = outputPath;
            BuildError           = null;
            ResultWarningDetails = new List<(string, List<int>)>();
            ResultFamilies       = new Dictionary<string, List<string>>();
            Done.Reset();
        }

        private const string BUILD_LOG = @"C:\RevitOutput\build_log.txt";

        public void Execute(UIApplication app)
        {
            // Pass log path so every warning/error is written immediately
            // as Revit raises it, including which element IDs are affected.
            var warningCollector = new WarningCollector(BUILD_LOG);
            try
            {
                var builder = new ModelBuilder(app.Application);
                builder.BuildModel(_transactionJson, _outputPath, warningCollector);
                ResultWarningDetails = warningCollector.WarningDetails;

                // Write post-build summary to build log
                warningCollector.LogSummary("BuildHandler summary");

                try { ResultFamilies = CollectFamilies(app.ActiveUIDocument.Document); }
                catch (Exception ex) { Console.WriteLine($"[BuildHandler] Family collection skipped: {ex.Message}"); }

                if (warningCollector.Warnings.Count > 0)
                    Console.WriteLine(
                        $"[BuildHandler] {warningCollector.Warnings.Count} warning(s): " +
                        string.Join(" | ", warningCollector.Warnings));
                if (warningCollector.Errors.Count > 0)
                    Console.WriteLine(
                        $"[BuildHandler] {warningCollector.Errors.Count} error(s): " +
                        string.Join(" | ", warningCollector.Errors));
            }
            catch (Exception ex)
            {
                BuildError           = ex;
                ResultWarningDetails = warningCollector.WarningDetails;
                Console.WriteLine($"[BuildHandler] Error: {ex}");
            }
            finally { Done.Set(); }
        }

        public string GetName() => "RevitModelBuilderAddin.BuildModel";

        private static Dictionary<string, List<string>> CollectFamilies(Document doc)
        {
            var result = new Dictionary<string, List<string>>();

            result["structural_columns"] = new FilteredElementCollector(doc)
                .OfCategory(BuiltInCategory.OST_StructuralColumns)
                .OfClass(typeof(FamilySymbol))
                .Cast<FamilySymbol>()
                .Select(fs => $"{fs.Family.Name}::{fs.Name}")
                .Distinct().OrderBy(s => s).ToList();

            result["wall_types"] = new FilteredElementCollector(doc)
                .OfClass(typeof(WallType)).Cast<WallType>()
                .Select(wt => wt.Name).OrderBy(s => s).ToList();

            result["door_families"] = new FilteredElementCollector(doc)
                .OfCategory(BuiltInCategory.OST_Doors)
                .OfClass(typeof(FamilySymbol))
                .Cast<FamilySymbol>()
                .Select(fs => $"{fs.Family.Name}::{fs.Name}")
                .Distinct().OrderBy(s => s).ToList();

            result["window_families"] = new FilteredElementCollector(doc)
                .OfCategory(BuiltInCategory.OST_Windows)
                .OfClass(typeof(FamilySymbol))
                .Cast<FamilySymbol>()
                .Select(fs => $"{fs.Family.Name}::{fs.Name}")
                .Distinct().OrderBy(s => s).ToList();

            return result;
        }
    }

    // =========================================================================
    // ModelBuilder — full BIM model creation (batch mode, unchanged from v1)
    // =========================================================================

    public class ModelBuilder
    {
        private readonly Application _app;
        private const double MM = 1.0 / 304.8;
        private static readonly string BUILD_LOG = @"C:\RevitOutput\build_log.txt";

        public ModelBuilder(Application app) { _app = app; }

        // ------------------------------------------------------------------
        // Template discovery — now public static so App can call it
        // ------------------------------------------------------------------

        public static string FindTemplate()
        {
            string templateDir = @"C:\ProgramData\Autodesk\RVT 2023\Templates";
            if (!Directory.Exists(templateDir)) return null;

            string[] candidates = Directory.GetFiles(templateDir, "*.rte",
                                                     SearchOption.TopDirectoryOnly);
            foreach (string prefer in new[]
            {
                "Default_M_ENU.rte", "Default_M_CHS.rte",
                "Default_M_CHT.rte", "Default_I_ENU.rte",
            })
            {
                string hit = candidates.FirstOrDefault(p =>
                    string.Equals(Path.GetFileName(p), prefer,
                                  StringComparison.OrdinalIgnoreCase));
                if (hit != null) return hit;
            }

            string metric = candidates.FirstOrDefault(p =>
                Path.GetFileName(p).StartsWith("Default_M_",
                                               StringComparison.OrdinalIgnoreCase));
            if (metric != null) return metric;

            return candidates.FirstOrDefault();
        }

        // ------------------------------------------------------------------
        // Entry point
        // ------------------------------------------------------------------

        private static void Log(string msg)
        {
            try { File.AppendAllText(BUILD_LOG, $"[{DateTime.Now:HH:mm:ss}] {msg}\r\n"); }
            catch { }
        }

        public void BuildModel(string transactionJson, string outputPath,
                               WarningCollector warningCollector = null)
        {
            try { Directory.CreateDirectory(@"C:\RevitOutput"); } catch { }
            File.WriteAllText(BUILD_LOG, $"[{DateTime.Now:HH:mm:ss}] BuildModel started.\r\n");

            Log("Deserialising transaction JSON...");
            var tx = JsonConvert.DeserializeObject<RevitTransaction>(
                         transactionJson,
                         new JsonSerializerSettings {
                             MissingMemberHandling = MissingMemberHandling.Ignore,
                             NullValueHandling     = NullValueHandling.Ignore,
                         })
                     ?? throw new Exception("Failed to deserialise transaction JSON.");
            Log($"JSON OK — levels:{tx.Levels?.Count} grids:{tx.Grids?.Count} " +
                $"columns:{tx.Columns?.Count} walls:{tx.Walls?.Count} " +
                $"framing:{tx.StructuralFraming?.Count} slabs:{tx.Slabs?.Count}");

            string template = FindTemplate();
            if (template == null)
                throw new Exception(
                    @"No .rte template found in C:\ProgramData\Autodesk\RVT 2023\Templates\.");
            Log($"Using template: {template}");

            Document doc = _app.NewProjectDocument(template);
            Log("Document created.");

            try
            {
                using (var t = new Transaction(doc, "Build BIM Model from Floor Plan"))
                {
                    if (warningCollector != null)
                    {
                        var failOpts = t.GetFailureHandlingOptions();
                        failOpts.SetFailuresPreprocessor(warningCollector);
                        t.SetFailureHandlingOptions(failOpts);
                    }

                    t.Start();
                    Log("CreateLevels...");           CreateLevels          (doc, tx.Levels);
                    Log("CreateGrids...");            CreateGrids           (doc, tx.Grids);
                    Log("CreateColumns...");          CreateColumns         (doc, tx.Columns);
                    Log("CreateWalls...");            CreateWalls           (doc, tx.Walls);
                    Log("CreateSlabs...");             CreateSlabs           (doc, tx.Slabs);
                    Log("CreateStructuralFraming..."); CreateStructuralFraming(doc, tx.StructuralFraming);
                    Log("Committing...");
                    t.Commit();
                    Log("Transaction committed.");
                }

                // ── Level cleanup: two separate transactions ──────────────────────
                // Transaction A: MOVE leftover template levels to safe far elevations.
                //   This must commit BEFORE Transaction B so that even if deletion
                //   fails and B rolls back, the levels are already out of the way.
                try
                {
                    using (var mt = new Transaction(doc, "Move Leftover Levels"))
                    {
                        if (warningCollector != null)
                        {
                            var mfo = mt.GetFailureHandlingOptions();
                            mfo.SetFailuresPreprocessor(warningCollector);
                            mt.SetFailureHandlingOptions(mfo);
                        }
                        mt.Start();
                        MoveLeftoverLevels(doc, tx.Levels);
                        mt.Commit();
                        Log("Leftover levels moved.");
                    }
                }
                catch (Exception mex) { Log($"Level move (non-fatal): {mex.Message}"); }

                // Transaction B: DELETE leftover levels (optional — ok if it fails,
                //   they are already far from the model after Transaction A).
                try
                {
                    using (var ct = new Transaction(doc, "Delete Leftover Levels"))
                    {
                        if (warningCollector != null)
                        {
                            var cfo = ct.GetFailureHandlingOptions();
                            cfo.SetFailuresPreprocessor(warningCollector);
                            ct.SetFailureHandlingOptions(cfo);
                        }
                        ct.Start();
                        DeleteLeftoverLevels(doc, tx.Levels);
                        ct.Commit();
                        Log("Leftover levels deleted.");
                    }
                }
                catch (Exception cex) { Log($"Level delete (non-fatal): {cex.Message}"); }

                try
                {
                    using (var vt = new Transaction(doc, "Create Views and Sheet"))
                    {
                        // Attach the same WarningCollector so view-creation warnings
                        // (e.g. duplicate view names, missing title blocks) are captured
                        // and dismissed — preventing Revit from showing blocking dialogs.
                        if (warningCollector != null)
                        {
                            var vfo = vt.GetFailureHandlingOptions();
                            vfo.SetFailuresPreprocessor(warningCollector);
                            vt.SetFailureHandlingOptions(vfo);
                        }
                        vt.Start();
                        CreateViewsAndSheet(doc);
                        vt.Commit();
                        Log("Views and sheet created.");
                    }
                }
                catch (Exception vex) { Log($"View/sheet creation (non-fatal): {vex.Message}"); }

                Log($"Saving to {outputPath} ...");
                var saveOpts = new SaveAsOptions { OverwriteExistingFile = true };
                doc.SaveAs(outputPath, saveOpts);
                Log("Save complete.");
                Console.WriteLine($"[ModelBuilder] Saved: {outputPath}");
            }
            finally { doc.Close(false); }
        }

        // ------------------------------------------------------------------
        // Views and sheet
        // ------------------------------------------------------------------

        private void CreateViewsAndSheet(Document doc)
        {
            Level level0 = GetLevel(doc, "Level 0");
            if (level0 == null) { Log("CreateViewsAndSheet: Level 0 not found — skipped."); return; }

            ViewFamilyType fpVFT = new FilteredElementCollector(doc)
                .OfClass(typeof(ViewFamilyType)).Cast<ViewFamilyType>()
                .FirstOrDefault(v => v.ViewFamily == ViewFamily.FloorPlan);

            ViewFamilyType cpVFT = new FilteredElementCollector(doc)
                .OfClass(typeof(ViewFamilyType)).Cast<ViewFamilyType>()
                .FirstOrDefault(v => v.ViewFamily == ViewFamily.CeilingPlan);

            ViewPlan floorPlan = null;
            if (fpVFT != null)
            {
                floorPlan      = ViewPlan.Create(doc, fpVFT.Id, level0.Id);
                floorPlan.Name = "Ground Floor Plan";
                Log("Floor plan view created for Level 0.");
            }
            if (cpVFT != null)
            {
                try
                {
                    var cp = ViewPlan.Create(doc, cpVFT.Id, level0.Id);
                    cp.Name = "Level 0 Ceiling Plan";
                    Log("Ceiling plan view created for Level 0.");
                }
                catch (Exception ex) { Log($"[WARN] Ceiling plan for Level 0: {ex.Message}"); }
            }

            // Collect our intended levels (those NOT moved to far elevations).
            // We only create views for levels at reasonable model elevations (< 200 m).
            var intendedLevels = new FilteredElementCollector(doc)
                .OfClass(typeof(Level)).Cast<Level>()
                .Where(l => l.Elevation * 304.8 < 200_000)   // < 200 m
                .OrderBy(l => l.Elevation)
                .ToList();

            foreach (var lvl in intendedLevels)
            {
                // Skip Level 0 — floor plan already created above.
                if (string.Equals(lvl.Name, "Level 0", StringComparison.OrdinalIgnoreCase))
                    continue;

                // ── Floor Plan view ──────────────────────────────────────────────
                if (fpVFT != null)
                {
                    bool hasFP = new FilteredElementCollector(doc)
                        .OfClass(typeof(ViewPlan)).Cast<ViewPlan>()
                        .Any(vp => vp.ViewType == ViewType.FloorPlan &&
                                   vp.GenLevel != null &&
                                   vp.GenLevel.Id == lvl.Id);

                    if (!hasFP)
                    {
                        try
                        {
                            ViewPlan vp = ViewPlan.Create(doc, fpVFT.Id, lvl.Id);
                            // Try to name it exactly as the level (e.g. "Level 1").
                            // Fall back with suffix if that name is already taken.
                            try { vp.Name = lvl.Name; }
                            catch { try { vp.Name = $"{lvl.Name} Floor Plan"; } catch { } }
                            Log($"Floor plan view created for '{lvl.Name}'.");
                        }
                        catch (Exception ex)
                        {
                            Log($"[WARN] Could not create floor plan view for '{lvl.Name}': {ex.Message}");
                        }
                    }
                }

                // ── Reflected Ceiling Plan view ──────────────────────────────────
                if (cpVFT != null)
                {
                    bool hasCP = new FilteredElementCollector(doc)
                        .OfClass(typeof(ViewPlan)).Cast<ViewPlan>()
                        .Any(vp => vp.ViewType == ViewType.CeilingPlan &&
                                   vp.GenLevel != null &&
                                   vp.GenLevel.Id == lvl.Id);

                    if (!hasCP)
                    {
                        try
                        {
                            var cp = ViewPlan.Create(doc, cpVFT.Id, lvl.Id);
                            try { cp.Name = $"{lvl.Name} Ceiling Plan"; } catch { }
                            Log($"Ceiling plan view created for '{lvl.Name}'.");
                        }
                        catch (Exception ex)
                        {
                            Log($"[WARN] Could not create ceiling plan view for '{lvl.Name}': {ex.Message}");
                        }
                    }
                }
            }

            ViewFamilyType tdVFT = new FilteredElementCollector(doc)
                .OfClass(typeof(ViewFamilyType)).Cast<ViewFamilyType>()
                .FirstOrDefault(v => v.ViewFamily == ViewFamily.ThreeDimensional);

            if (tdVFT != null)
            {
                View3D v3d = View3D.CreateIsometric(doc, tdVFT.Id);
                v3d.Name   = "3D Overview";
                Log("3D view created.");
            }

            FamilySymbol titleBlock = new FilteredElementCollector(doc)
                .OfClass(typeof(FamilySymbol)).Cast<FamilySymbol>()
                .FirstOrDefault(s => s.Category?.Id?.IntegerValue ==
                                     (int)BuiltInCategory.OST_TitleBlocks);

            if (titleBlock == null) { Log("No title block found — sheet skipped."); return; }
            if (!titleBlock.IsActive) titleBlock.Activate();

            ViewSheet sheet = ViewSheet.Create(doc, titleBlock.Id);
            sheet.Name        = "Ground Floor — Auto Generated";
            sheet.SheetNumber = "A-001";
            Log("Sheet A-001 created.");

            if (floorPlan != null &&
                Viewport.CanAddViewToSheet(doc, sheet.Id, floorPlan.Id))
            {
                BoundingBoxUV outline = sheet.Outline;
                double u = (outline.Min.U + outline.Max.U) / 2.0;
                double v = (outline.Min.V + outline.Max.V) / 2.0;
                Viewport.Create(doc, sheet.Id, floorPlan.Id, new XYZ(u, v, 0));
                Log("Floor plan placed on sheet A-001.");
            }
        }

        // ------------------------------------------------------------------
        // 1. Levels
        // ------------------------------------------------------------------

        private void CreateLevels(Document doc, List<LevelItem> levels)
        {
            var effective = (levels != null && levels.Count >= 2)
                ? levels
                : new List<LevelItem> {
                      new LevelItem { Name = "Level 0", Elevation = 0    },
                      new LevelItem { Name = "Level 1", Elevation = 3000 },
                  };

            var allLevels = new FilteredElementCollector(doc)
                .OfClass(typeof(Level)).Cast<Level>().ToList();

            foreach (var lv in effective)
            {
                double targetElevFt = lv.Elevation * MM;

                // If a level with this name already exists, move it to the
                // correct elevation rather than skipping it.  Templates often
                // have "Level 1" / "Level 2" (or even "Level 0" / "Level 1")
                // at arbitrary heights — we must own the elevation.
                Level named = allLevels.FirstOrDefault(x =>
                    string.Equals(x.Name, lv.Name, StringComparison.OrdinalIgnoreCase));
                if (named != null)
                {
                    if (Math.Abs(named.Elevation - targetElevFt) > 1e-4)
                    {
                        named.Elevation = targetElevFt;
                        Log($"Level '{lv.Name}' existed — moved to {lv.Elevation:F0} mm.");
                    }
                    else
                        Log($"Level '{lv.Name}' already at {lv.Elevation:F0} mm — OK.");
                    continue;
                }

                // No level with this name — check if one already sits at the
                // target elevation and re-use it (rename).
                Level atElev = allLevels.FirstOrDefault(x =>
                    Math.Abs(x.Elevation - targetElevFt) < 1e-4);
                if (atElev != null)
                {
                    try { atElev.Name = lv.Name; } catch { }
                    Log($"Renamed existing level to '{lv.Name}' at {lv.Elevation:F0} mm.");
                    continue;
                }

                Level created = Level.Create(doc, targetElevFt);
                created.Name = lv.Name;
                allLevels.Add(created);
                Log($"Created level '{lv.Name}' at {lv.Elevation:F0} mm.");
            }

            // NOTE: Leftover template level deletion is intentionally NOT done
            // here.  It runs in a separate transaction (DeleteLeftoverLevels)
            // AFTER the main build transaction commits, so a deletion failure
            // cannot contaminate column / grid placements in this transaction.
        }

        /// <summary>
        /// Move every level NOT in the intended list to safe far-away elevations.
        /// Called in its own committed transaction BEFORE DeleteLeftoverLevels so
        /// that even if deletion fails and its transaction rolls back, the levels
        /// are already out of the model.
        /// </summary>
        private void MoveLeftoverLevels(Document doc, List<LevelItem> levels)
        {
            var effective = (levels != null && levels.Count >= 2)
                ? levels
                : new List<LevelItem> {
                      new LevelItem { Name = "Level 0", Elevation = 0    },
                      new LevelItem { Name = "Level 1", Elevation = 3000 },
                  };

            var intendedNames = new HashSet<string>(
                effective.Select(l => l.Name), StringComparer.OrdinalIgnoreCase);

            double maxIntendedElevMm = effective.Max(l => l.Elevation);

            var leftover = new FilteredElementCollector(doc)
                .OfClass(typeof(Level)).Cast<Level>()
                .Where(l => !intendedNames.Contains(l.Name))
                .OrderBy(l => l.Elevation)
                .ToList();

            int k = 1;
            foreach (var lvl in leftover)
            {
                double targetMm = maxIntendedElevMm + k * 3000.0;
                try
                {
                    lvl.Elevation = targetMm * MM;
                    Log($"Moved leftover level '{lvl.Name}' to {targetMm:F0} mm.");
                    k++;
                }
                catch (Exception ex)
                {
                    Log($"[WARN] Could not move level '{lvl.Name}': {ex.Message}");
                }
            }
        }

        /// <summary>
        /// Delete Revit template levels whose names are NOT in our intended level list.
        /// MoveLeftoverLevels MUST have been committed first — levels are already far
        /// from the model, so even if deletion fails the model is unaffected.
        /// </summary>
        private void DeleteLeftoverLevels(Document doc, List<LevelItem> levels)
        {
            var effective = (levels != null && levels.Count >= 2)
                ? levels
                : new List<LevelItem> {
                      new LevelItem { Name = "Level 0", Elevation = 0    },
                      new LevelItem { Name = "Level 1", Elevation = 3000 },
                  };

            var intendedNames = new HashSet<string>(
                effective.Select(l => l.Name), StringComparer.OrdinalIgnoreCase);

            var leftover = new FilteredElementCollector(doc)
                .OfClass(typeof(Level)).Cast<Level>()
                .Where(l => !intendedNames.Contains(l.Name))
                .ToList();

            foreach (var lvl in leftover)
            {
                // Delete associated floor/ceiling plan views first.
                var viewIds = new FilteredElementCollector(doc)
                    .OfClass(typeof(ViewPlan)).Cast<ViewPlan>()
                    .Where(vp => vp.GenLevel != null && vp.GenLevel.Id == lvl.Id)
                    .Select(vp => vp.Id)
                    .ToList();
                foreach (var vid in viewIds)
                    try { doc.Delete(vid); } catch { }

                try
                {
                    doc.Delete(lvl.Id);
                    Log($"Deleted leftover level '{lvl.Name}'.");
                }
                catch (Exception ex)
                {
                    Log($"[INFO] Could not delete leftover level '{lvl.Name}' (already moved to safe elevation): {ex.Message}");
                }
            }
        }

        // ------------------------------------------------------------------
        // 2. Grids
        // ------------------------------------------------------------------

        private void CreateGrids(Document doc, List<GridItem> grids)
        {
            if (grids == null) return;
            foreach (var g in grids)
            {
                if (g.Start == null || g.End == null) continue;
                XYZ start = Pt(g.Start); XYZ end = Pt(g.End);
                if (start.DistanceTo(end) < 1e-9) continue;
                Grid grid = Grid.Create(doc, Line.CreateBound(start, end));
                if (!string.IsNullOrWhiteSpace(g.Name))
                    try { grid.Name = g.Name; } catch { }
            }
        }

        // ------------------------------------------------------------------
        // 3. Structural columns
        // ------------------------------------------------------------------

        private void CreateColumns(Document doc, List<ColumnItem> columns)
        {
            if (columns == null || columns.Count == 0) return;

            var (rectSym, roundSym) = LoadConcreteColumnFamilies(doc);
            if (rectSym == null && roundSym == null)
            {
                Log("[WARN] No structural column family found — columns skipped.");
                return;
            }

            // Activate symbols that were found
            try { if (rectSym  != null && !rectSym.IsActive)  rectSym.Activate(); }
            catch (Exception ex) { Log($"[WARN] rectSym.Activate() failed: {ex.Message}"); rectSym = null; }
            try { if (roundSym != null && !roundSym.IsActive) roundSym.Activate(); }
            catch (Exception ex) { Log($"[WARN] roundSym.Activate() failed: {ex.Message}"); roundSym = null; }

            // Revit API requires Regenerate() after Activate() before any family
            // instance can be placed — without it NewFamilyInstance throws
            // "The referenced object is not valid".
            doc.Regenerate();

            if (rectSym == null && roundSym == null)
            {
                Log("[WARN] Both column symbols failed activation — columns skipped.");
                return;
            }

            // Log families and their parameter names
            if (rectSym  != null) { Log($"Rect column family: '{rectSym.FamilyName}', base type: '{rectSym.Name}'");   LogEditableDoubleParams(rectSym); }
            if (roundSym != null) { Log($"Round column family: '{roundSym.FamilyName}', base type: '{roundSym.Name}'"); LogEditableDoubleParams(roundSym); }

            int placed = 0, skipped = 0;
            foreach (var col in columns)
            {
                if (col.Location == null)
                {
                    Log($"Column {col.Id}: no location data — skipped.");
                    skipped++;
                    continue;
                }

                // Enforce minimum size (200 mm).
                double w = Math.Max(col.Width,  200.0);
                double d = Math.Max(col.Depth,  200.0);
                if (w != col.Width || d != col.Depth)
                    Log($"Column {col.Id}: size clamped from {col.Width:F0}×{col.Depth:F0} to {w:F0}×{d:F0}mm");

                Level baseLevel = GetLevel(doc, col.Level ?? "Level 0");
                if (baseLevel == null)
                {
                    Log($"Column {col.Id}: base level '{col.Level}' not found — skipped.");
                    skipped++;
                    continue;
                }

                // Pick rect vs round family, with fallback
                bool useRound = IsRoundColumn(col);
                FamilySymbol baseSymbol;
                if (useRound)
                {
                    baseSymbol = roundSym ?? rectSym;
                    Log($"Column {col.Id}: shape=round → family '{baseSymbol?.FamilyName}'" +
                        (roundSym == null ? " (round not loaded, using rect fallback)" : ""));
                }
                else
                {
                    baseSymbol = rectSym ?? roundSym;
                    Log($"Column {col.Id}: shape=rect → family '{baseSymbol?.FamilyName}'" +
                        (rectSym == null ? " (rect not loaded, using round fallback)" : ""));
                }

                if (baseSymbol == null)
                {
                    Log($"Column {col.Id}: no family symbol available — skipped.");
                    skipped++;
                    continue;
                }

                // Get or create a properly-sized family type.
                FamilySymbol symbol = GetOrDuplicateColumnType(doc, baseSymbol, w, d);
                if (symbol == null)
                {
                    Log($"Column {col.Id}: type creation failed — falling back to base symbol " +
                        $"'{baseSymbol.Name}' (dimensions may be wrong).");
                    symbol = baseSymbol;
                }
                if (!symbol.IsActive) symbol.Activate();
                // Regenerate after per-column symbol activation so Revit resolves
                // the reference before NewFamilyInstance is called.
                doc.Regenerate();

                try
                {
                    XYZ loc = Pt(col.Location);
                    FamilyInstance inst = doc.Create.NewFamilyInstance(
                        loc, symbol, baseLevel, StructuralType.Column);

                    if (inst == null)
                    {
                        Log($"Column {col.Id}: NewFamilyInstance returned null.");
                        skipped++;
                        continue;
                    }

                    // Set top-level constraint so column spans floor-to-floor.
                    Level topLevel = GetLevel(doc, col.TopLevel ?? "Level 1");
                    if (topLevel != null && topLevel.Elevation <= baseLevel.Elevation + 1e-6)
                    {
                        Log($"Column {col.Id}: top level '{col.TopLevel}' is at or below base " +
                            $"({topLevel.Elevation * 304.8:F0} mm ≤ {baseLevel.Elevation * 304.8:F0} mm) " +
                            $"— searching for next level above.");
                        topLevel = GetLevelAbove(doc, baseLevel);
                    }
                    if (topLevel != null)
                        inst.get_Parameter(BuiltInParameter.FAMILY_TOP_LEVEL_PARAM)?.Set(topLevel.Id);
                    else
                        Log($"Column {col.Id}: no level above base found — column height left at family default.");

                    placed++;
                }
                catch (Exception ex)
                {
                    Log($"Column {col.Id}: placement exception — {ex.Message}");
                    skipped++;
                }
            }

            Log($"Columns complete: {placed} placed, {skipped} skipped out of {columns.Count} total.");
        }

        /// <summary>
        /// Log all editable double parameters on a FamilySymbol.
        /// Writes to build_log.txt so we know exactly which param names the loaded family uses.
        /// </summary>
        private void LogEditableDoubleParams(FamilySymbol sym)
        {
            try
            {
                var parts = sym.Parameters.Cast<Parameter>()
                    .Where(p => !p.IsReadOnly && p.StorageType == StorageType.Double)
                    .Select(p =>
                    {
                        double mm = UnitUtils.ConvertFromInternalUnits(
                            p.AsDouble(), UnitTypeId.Millimeters);
                        return $"'{p.Definition.Name}'={mm:F0}mm";
                    })
                    .ToList();
                Log($"  Editable dimension params: {(parts.Count > 0 ? string.Join(", ", parts) : "(none)")}");
            }
            catch (Exception ex) { Log($"  LogEditableDoubleParams error: {ex.Message}"); }
        }

        /// <summary>
        /// Returns true if the family name looks like a UC/Universal Column (steel I-beam).
        /// We want to skip these and prefer CJY concrete families.
        /// </summary>
        private static bool IsUCFamily(string name)
        {
            if (string.IsNullOrEmpty(name)) return false;
            string n = name.ToUpperInvariant();
            bool hasUniversal = n.Contains("UNIVERSAL");
            bool hasColumn    = n.Contains("COLUMN");
            bool startsWithUC = n.StartsWith("UC-") || n.StartsWith("UC_");
            return (hasUniversal && hasColumn) || startsWithUC;
        }

        /// <summary>
        /// Returns true if the column should use the round (RC Round) family symbol.
        /// Priority: explicit shape field → type_mark keywords.
        /// </summary>
        private static bool IsRoundColumn(ColumnItem col)
        {
            // 1. Explicit shape field from Python (highest priority)
            if (!string.IsNullOrEmpty(col.Shape))
            {
                string shp = col.Shape.ToUpperInvariant();
                // "circular", "round", "circ" → round family
                if (shp.Contains("CIRC") || shp.Contains("ROUND"))
                    return true;
                // "rectangular", "rect", "square", "sq" → rectangular family
                // A square (equal width×depth) column is still a RECTANGULAR column.
                if (shp.Contains("RECT") || shp.Contains("SQUARE") || shp.Contains("SQ"))
                    return false;
            }

            // 2. Type mark explicitly says "Round" or "Circ"
            string tm = col.TypeMark ?? "";
            if (tm.IndexOf("Round", StringComparison.OrdinalIgnoreCase) >= 0 ||
                tm.IndexOf("Circ",  StringComparison.OrdinalIgnoreCase) >= 0)
                return true;

            // Default: assume rectangular.
            // NOTE: equal width×depth (800×800, 600×600 etc.) does NOT mean circular —
            // those are square concrete columns which use the rectangular family.
            return false;
        }

        /// <summary>
        /// Returns true if the family is a concrete column (not steel I-beam).
        /// Used to filter families after loading from disk.
        /// </summary>
        private static bool IsConcreteColumnFamily(Family fam)
        {
            if (fam == null) return false;
            string name = fam.Name.ToUpperInvariant();

            // Skip steel I-beams
            if (name.Contains("UC-") || name.Contains("UB-") ||
                name.Contains("UNIVERSAL") || name.Contains("I-BEAM") ||
                name.Contains("WIDE FLANGE") || name.Contains("W-SHAPE"))
                return false;

            // Must contain concrete-related terms
            bool hasConcrete = name.Contains("CONCRETE") || name.Contains("RC ") ||
                               name.Contains("CJY") || name.Contains("CAST-IN-PLACE");

            // Must be a column (the category is structural column, but we also check name)
            bool isColumn = name.Contains("COLUMN") || name.Contains("RECTANGULAR") ||
                            name.Contains("ROUND") || name.Contains("CIRCULAR");

            return hasConcrete && isColumn;
        }

        /// <summary>
        /// Load or find the CJY concrete column families (rectangular and round).
        /// Search priority per family type:
        ///   1. Already in doc: prefer "CJY" families; skip UC/Universal entirely.
        ///   2. Custom folder C:\MyDocuments\3. Revit Family Files (recursive).
        ///   3. Autodesk library fallback (*Concrete*Column*.rfa, metric-first).
        /// Returns (rectSymbol, roundSymbol) — either may be null if not found.
        /// </summary>
        private (FamilySymbol rect, FamilySymbol round) LoadConcreteColumnFamilies(Document doc)
        {
            // ── DEBUG: Check custom folder existence ───────────────────────
            string testFolder = @"C:\MyDocument\3. Revit Family Files\1. column";
            if (Directory.Exists(testFolder))
            {
                var files = Directory.GetFiles(testFolder, "*.rfa", SearchOption.AllDirectories);
                Log($"Found {files.Length} RFA files in {testFolder}:");
                foreach (var f in files.Take(10)) // show first 10
                    Log($"  - {Path.GetFileName(f)}");
            }
            else
            {
                Log($"Folder NOT FOUND: {testFolder}");
            }

            // ── Step 1: scan symbols already in the document ──────────────────
            var inDoc = new FilteredElementCollector(doc)
                .OfClass(typeof(FamilySymbol)).Cast<FamilySymbol>()
                .Where(s => s.Category?.Id?.IntegerValue ==
                            (int)BuiltInCategory.OST_StructuralColumns)
                .ToList();

            FamilySymbol rectSym  = null;
            FamilySymbol roundSym = null;

            foreach (var s in inDoc)
            {
                if (IsUCFamily(s.FamilyName)) continue;   // skip steel I-beams
                // Get the family to check if it's concrete
                Family fam = s.Family;
                if (fam == null) continue;
                if (!IsConcreteColumnFamily(fam))
                {
                    Log($"Skipping non-concrete family: '{s.FamilyName}'");
                    continue;
                }

                string fn = s.FamilyName.ToUpperInvariant();
                bool isCJY   = fn.Contains("CJY");
                bool isRound = fn.Contains("ROUND") || fn.Contains("CIRCULAR") ||
                               s.Name.ToUpperInvariant().Contains("ROUND");
                bool isRect  = !isRound;

                if (isCJY)
                {
                    if (isRound  && roundSym == null) { roundSym = s; Log($"Using preferred CJY round family in doc: '{s.FamilyName}' / '{s.Name}'"); }
                    if (isRect   && rectSym  == null) { rectSym  = s; Log($"Using preferred CJY rect family in doc: '{s.FamilyName}' / '{s.Name}'"); }
                }
                else
                {
                    // Non-CJY concrete fallback from doc (used only if CJY not found)
                    if (isRound && roundSym == null) roundSym = s;
                    if (isRect  && rectSym  == null) rectSym  = s;
                }
            }

            if (rectSym != null && roundSym != null)
            {
                Log($"Both column families found in document. Rect='{rectSym.FamilyName}', Round='{roundSym.FamilyName}'");
                return (rectSym, roundSym);
            }

            // ── Step 2: search custom folder ──────────────────────────────────
            // Priority patterns (tried in order): CJY-specific first, then general.
            // "CJY_Concrete-Rectangular-Column.rfa" — has "Concrete" and "Column"
            // but "Rectangular" not "Rectangle", so must use broad pattern.
            var rectPatterns  = new[] { "*CJY*Column*.rfa",  "*Concrete*Rectangular*Column*.rfa", "*Concrete*Column*.rfa" };
            var roundPatterns = new[] { "*CJY*Round*.rfa",   "*RC*Round*Column*.rfa",              "*Concrete*Round*Column*.rfa" };

            var customFolders = new[] {
                // Both spellings tried — "MyDocument" (no s) and "MyDocuments"
                @"C:\MyDocument\3. Revit Family Files\1. column", // user-confirmed path
                @"C:\MyDocument\3. Revit Family Files",
                @"C:\MyDocument\3. Revit Family\1. column",
                @"C:\MyDocument\3. Revit Family",
                @"C:\MyDocuments\3. Revit Family\1. column",
                @"C:\MyDocuments\3. Revit Family",
                @"C:\MyDocuments\3. Revit Family Files",
                @"C:\Users\Public\Documents\3. Revit Family Files",
            };

            foreach (var folder in customFolders)
            {
                if (!Directory.Exists(folder)) continue;
                if (rectSym != null && roundSym != null) break;

                if (rectSym == null)
                {
                    foreach (var pattern in rectPatterns)
                    {
                        string[] candidates;
                        try { candidates = Directory.GetFiles(folder, pattern, SearchOption.AllDirectories); }
                        catch { candidates = new string[0]; }
                        foreach (var path in candidates)
                        {
                            if (IsUCFamily(Path.GetFileName(path))) continue;
                            try
                            {
                                if (doc.LoadFamily(path, out Family fam) && fam != null)
                                {
                                    if (!IsConcreteColumnFamily(fam))
                                    {
                                        Log($"Skipping non-concrete family from custom folder: {Path.GetFileName(path)}");
                                        // Optionally delete the loaded family to keep document clean
                                        try { doc.Delete(fam.Id); } catch { }
                                        continue;
                                    }
                                    var sym = fam.GetFamilySymbolIds()
                                                 .Select(id => doc.GetElement(id) as FamilySymbol)
                                                 .FirstOrDefault(s => s != null);
                                    if (sym != null)
                                    {
                                        rectSym = sym;
                                        Log($"Loaded rect column family from custom folder: {path}");
                                        break;
                                    }
                                }
                            }
                            catch { }
                        }
                        if (rectSym != null) break;
                    }
                }

                if (roundSym == null)
                {
                    foreach (var pattern in roundPatterns)
                    {
                        string[] candidates;
                        try { candidates = Directory.GetFiles(folder, pattern, SearchOption.AllDirectories); }
                        catch { candidates = new string[0]; }
                        foreach (var path in candidates)
                        {
                            if (IsUCFamily(Path.GetFileName(path))) continue;
                            try
                            {
                                if (doc.LoadFamily(path, out Family fam) && fam != null)
                                {
                                    if (!IsConcreteColumnFamily(fam))
                                    {
                                        Log($"Skipping non-concrete family from custom folder: {Path.GetFileName(path)}");
                                        try { doc.Delete(fam.Id); } catch { }
                                        continue;
                                    }
                                    var sym = fam.GetFamilySymbolIds()
                                                 .Select(id => doc.GetElement(id) as FamilySymbol)
                                                 .FirstOrDefault(s => s != null);
                                    if (sym != null)
                                    {
                                        roundSym = sym;
                                        Log($"Loaded round column family from custom folder: {path}");
                                        break;
                                    }
                                }
                            }
                            catch { }
                        }
                        if (roundSym != null) break;
                    }
                }
            }

            // ── Step 3: Autodesk library fallback ────────────────────────────
            if (rectSym == null || roundSym == null)
            {
                var searchRoots = new[] {
                    Path.Combine(Environment.GetFolderPath(
                        Environment.SpecialFolder.CommonApplicationData),
                        @"Autodesk\RVT 2023\Libraries"),
                    @"C:\ProgramData\Autodesk\RVT 2023\Libraries",
                    @"D:\ProgramData\Autodesk\RVT 2023\Libraries",
                };

                foreach (var root in searchRoots)
                {
                    if (!Directory.Exists(root)) continue;
                    string[] rfas;
                    try { rfas = Directory.GetFiles(root, "*Concrete*Column*.rfa", SearchOption.AllDirectories); }
                    catch { continue; }

                    // Metric-first
                    foreach (var path in rfas.OrderBy(p =>
                        Path.GetFileName(p).StartsWith("M_", StringComparison.OrdinalIgnoreCase) ? 0 : 1))
                    {
                        if (IsUCFamily(Path.GetFileName(path))) continue;
                        if (rectSym != null && roundSym != null) break;
                        try
                        {
                            if (doc.LoadFamily(path, out Family fam) && fam != null)
                            {
                                if (!IsConcreteColumnFamily(fam))
                                {
                                    Log($"Skipping non-concrete family from Autodesk lib: {Path.GetFileName(path)}");
                                    try { doc.Delete(fam.Id); } catch { }
                                    continue;
                                }
                                string fn = (fam.Name ?? "").ToUpperInvariant();
                                bool isRound = fn.Contains("ROUND") || fn.Contains("CIRCULAR");
                                var sym = fam.GetFamilySymbolIds()
                                             .Select(id => doc.GetElement(id) as FamilySymbol)
                                             .FirstOrDefault(s => s != null);
                                if (sym != null)
                                {
                                    if (isRound && roundSym == null) { roundSym = sym; Log($"Loaded round column family (Autodesk lib): {path}"); }
                                    else if (!isRound && rectSym == null) { rectSym = sym; Log($"Loaded rect column family (Autodesk lib): {path}"); }
                                    else
                                    {
                                        // We already have both, delete this extra family
                                        try { doc.Delete(fam.Id); } catch { }
                                    }
                                }
                            }
                        }
                        catch { }
                    }
                    if (rectSym != null && roundSym != null) break;
                }
            }

            if (rectSym == null && roundSym == null)
                Log("[ERROR] Could not find or load any structural column family.");
            else if (rectSym == null)
                Log("[WARN] Only round column family found — rect columns will fall back to round.");
            else if (roundSym == null)
                Log("[WARN] Only rect column family found — round columns will fall back to rect.");

            return (rectSym, roundSym);
        }

        /// <summary>
        /// Find or create a FamilySymbol with the requested width × depth.
        /// Steps:
        ///   1. Look for an existing type with that name.
        ///   2. Duplicate the base symbol and set width/depth via known param names.
        ///   3. If named params fail, fall back to parameter discovery (scan all editable doubles).
        ///   4. Log every step — return null only if the duplicate call itself throws.
        /// </summary>
        private FamilySymbol GetOrDuplicateColumnType(
            Document doc, FamilySymbol baseSymbol, double wMm, double dMm)
        {
            // Round to nearest 50 mm so we reuse types across similar-sized columns.
            double w = Math.Round(wMm / 50.0) * 50;
            double d = Math.Round(dMm / 50.0) * 50;
            string typeName = $"{(int)w}x{(int)d}mm";

            // Re-use existing type if available.
            var found = new FilteredElementCollector(doc)
                .OfClass(typeof(FamilySymbol)).Cast<FamilySymbol>()
                .FirstOrDefault(s =>
                    s.FamilyName == baseSymbol.FamilyName &&
                    string.Equals(s.Name, typeName, StringComparison.OrdinalIgnoreCase));
            if (found != null)
            {
                Log($"  Reusing existing type '{typeName}'.");
                return found;
            }

            try
            {
                var newType = baseSymbol.Duplicate(typeName) as FamilySymbol;
                // Revit API: Regenerate after Duplicate so parameter accessors
                // on the new type work correctly before we set dimensions.
                doc.Regenerate();
                if (newType == null)
                {
                    Log($"  [ERROR] Duplicate('{typeName}') returned null.");
                    return null;
                }

                // ── Pass 1: try known parameter name pairs ─────────────────
                // Width candidates (largest dimension or "b" in Revit convention).
                string[] widthNames = { "b", "Width", "w", "Section Width", "Col Width", "Breadth", "B" };
                // Depth candidates ("h" in Revit convention).
                string[] depthNames = { "h", "Depth", "d", "Section Depth", "Col Depth", "H", "D" };

                bool widthSet = TrySetDimParam(newType, w, widthNames);
                bool depthSet = TrySetDimParam(newType, d, depthNames);

                if (!widthSet || !depthSet)
                {
                    // ── Pass 2: discovery — scan all editable doubles ──────
                    Log($"  Named params not matched (widthSet={widthSet}, depthSet={depthSet}) " +
                        $"for type '{typeName}'. Running discovery...");
                    DiscoverAndSetDimensions(newType, w, d, widthSet, depthSet);
                }

                // Verify what was actually set.
                LogEditableDoubleParams(newType);
                return newType;
            }
            catch (Exception ex)
            {
                Log($"  [ERROR] GetOrDuplicateColumnType '{typeName}': {ex.Message}");
                return null;
            }
        }

        /// <summary>
        /// Try each candidate parameter name in order; set the first match and return true.
        /// </summary>
        private bool TrySetDimParam(FamilySymbol sym, double valueMm, params string[] names)
        {
            foreach (var name in names)
            {
                var p = sym.LookupParameter(name);
                if (p != null && !p.IsReadOnly && p.StorageType == StorageType.Double)
                {
                    p.Set(valueMm / 304.8);
                    Log($"    Set '{name}' = {valueMm:F0}mm");
                    return true;
                }
            }
            return false;
        }

        /// <summary>
        /// Last-resort: enumerate all editable double parameters and assign width/depth
        /// to the two params whose names most resemble dimension fields.
        /// Only sets the dimension(s) that were not already set in Pass 1.
        /// </summary>
        private void DiscoverAndSetDimensions(
            FamilySymbol sym, double wMm, double dMm, bool widthAlreadySet, bool depthAlreadySet)
        {
            var editableDoubles = sym.Parameters.Cast<Parameter>()
                .Where(p => !p.IsReadOnly && p.StorageType == StorageType.Double)
                .ToList();

            if (editableDoubles.Count == 0)
            {
                Log("    Discovery: no editable double parameters found.");
                return;
            }

            Log($"    Discovery: found {editableDoubles.Count} editable double param(s): " +
                string.Join(", ", editableDoubles.Select(p => $"'{p.Definition.Name}'")));

            // Score each param: higher = more likely to be a section dimension.
            Parameter PickBest(string[] keywords, Parameter exclude)
            {
                return editableDoubles
                    .Where(p => p != exclude)
                    .OrderByDescending(p =>
                    {
                        string n = p.Definition.Name.ToLower();
                        return keywords.Sum(k => n.Contains(k) ? 2 : 0)
                             + (n.Length <= 2 ? 1 : 0); // short names like "b", "h" score higher
                    })
                    .FirstOrDefault();
            }

            Parameter widthParam = null;
            Parameter depthParam = null;

            if (!widthAlreadySet)
            {
                widthParam = PickBest(
                    new[] { "width", "breadth", "b", "w", "section" }, null);
                if (widthParam != null)
                {
                    widthParam.Set(wMm / 304.8);
                    Log($"    Discovery: set width via '{widthParam.Definition.Name}' = {wMm:F0}mm");
                }
                else Log("    Discovery: could not identify a width parameter.");
            }

            if (!depthAlreadySet)
            {
                depthParam = PickBest(
                    new[] { "depth", "h", "d", "height", "section" }, widthParam);
                if (depthParam != null)
                {
                    depthParam.Set(dMm / 304.8);
                    Log($"    Discovery: set depth via '{depthParam.Definition.Name}' = {dMm:F0}mm");
                }
                else Log("    Discovery: could not identify a depth parameter.");
            }
        }

        private static void SetParamMm(FamilySymbol sym, string paramName, double valueMm)
        {
            var p = sym.LookupParameter(paramName);
            if (p != null && !p.IsReadOnly && p.StorageType == StorageType.Double)
                p.Set(valueMm / 304.8);
        }

        // ------------------------------------------------------------------
        // 4. Walls
        // ------------------------------------------------------------------

        // ------------------------------------------------------------------
        // 4b. Structural Framing (beams)
        // ------------------------------------------------------------------

        private void CreateStructuralFraming(Document doc, List<FramingItem> framings)
        {
            if (framings == null || framings.Count == 0) return;

            // Prefer a CJY_-branded concrete beam family; fall back to any StructuralFraming symbol.
            FamilySymbol beamSym = new FilteredElementCollector(doc)
                .OfClass(typeof(FamilySymbol)).Cast<FamilySymbol>()
                .Where(s => s.Category?.Id?.IntegerValue ==
                            (int)BuiltInCategory.OST_StructuralFraming)
                .OrderByDescending(s => s.FamilyName.StartsWith("CJY_",
                    StringComparison.OrdinalIgnoreCase))
                .FirstOrDefault();

            if (beamSym == null)
            {
                Log("[WARN] No structural framing family found — beams skipped.");
                return;
            }
            if (!beamSym.IsActive)
            {
                try { beamSym.Activate(); doc.Regenerate(); }
                catch (Exception ex)
                {
                    Log($"[WARN] Beam symbol activation failed: {ex.Message}");
                    return;
                }
            }

            int placed = 0, skipped = 0;
            foreach (var beam in framings)
            {
                if (beam.StartPoint == null || beam.EndPoint == null) { skipped++; continue; }

                XYZ start = Pt(beam.StartPoint);
                XYZ end   = Pt(beam.EndPoint);
                if (start.DistanceTo(end) < 1e-6) { skipped++; continue; }

                Level level = GetLevel(doc, beam.Level ?? "Level 0");
                if (level == null) { skipped++; continue; }

                try
                {
                    var line = Line.CreateBound(start, end);
                    doc.Create.NewFamilyInstance(
                        line, beamSym, level, StructuralType.Beam);
                    placed++;
                }
                catch (Exception ex)
                {
                    Log($"[WARN] Beam {beam.Id} placement failed: {ex.Message}");
                    skipped++;
                }
            }
            Log($"Beams: {placed} placed, {skipped} skipped.");
        }

        private void CreateWalls(Document doc, List<WallItem> walls)
        {
            if (walls == null || walls.Count == 0) return;
            WallType defaultType = new FilteredElementCollector(doc)
                .OfClass(typeof(WallType)).Cast<WallType>().FirstOrDefault();
            if (defaultType == null) return;

            foreach (var w in walls)
            {
                if (w.StartPoint == null || w.EndPoint == null) continue;
                XYZ start = Pt(w.StartPoint); XYZ end = Pt(w.EndPoint);
                if (start.DistanceTo(end) < 1e-9) continue;
                Level baseLevel = GetLevel(doc, w.Level ?? "Level 0");
                if (baseLevel == null) continue;
                double height = (w.Height > 0 ? w.Height : 2800) * MM;
                Wall.Create(doc, Line.CreateBound(start, end),
                            defaultType.Id, baseLevel.Id, height, 0.0, false, w.IsStructural);
            }
        }

        // ------------------------------------------------------------------
        // 5. Floors
        // ------------------------------------------------------------------

        private void CreateSlabs(Document doc, List<SlabItem> slabs)
        {
            if (slabs == null || slabs.Count == 0) return;
            FloorType defaultFloorType = new FilteredElementCollector(doc)
                .OfClass(typeof(FloorType)).Cast<FloorType>().FirstOrDefault();
            if (defaultFloorType == null) return;

            foreach (var sl in slabs)
            {
                if (sl.BoundaryPoints == null || sl.BoundaryPoints.Count < 3) continue;
                Level level = GetLevel(doc, sl.Level ?? "Level 0");
                if (level == null) continue;

                var loop = new CurveLoop();
                int n = sl.BoundaryPoints.Count;
                for (int i = 0; i < n; i++)
                {
                    var p1 = sl.BoundaryPoints[i];
                    var p2 = sl.BoundaryPoints[(i + 1) % n];
                    loop.Append(Line.CreateBound(
                        new XYZ(p1.X * MM, p1.Y * MM, sl.Elevation * MM),
                        new XYZ(p2.X * MM, p2.Y * MM, sl.Elevation * MM)));
                }
                Floor.Create(doc, new List<CurveLoop> { loop }, defaultFloorType.Id, level.Id);
            }
        }

        // ------------------------------------------------------------------
        // Geometry helpers
        // ------------------------------------------------------------------

        private static XYZ Pt(PointData p) =>
            new XYZ(p.X / 304.8, p.Y / 304.8, p.Z / 304.8);

        private static Level GetLevel(Document doc, string name) =>
            new FilteredElementCollector(doc).OfClass(typeof(Level)).Cast<Level>()
                .FirstOrDefault(l =>
                    string.Equals(l.Name, name, StringComparison.OrdinalIgnoreCase));

        private static Level GetLevelAbove(Document doc, Level baseLevel) =>
            new FilteredElementCollector(doc).OfClass(typeof(Level)).Cast<Level>()
                .Where(l => l.Elevation > baseLevel.Elevation + 1e-6)
                .OrderBy(l => l.Elevation).FirstOrDefault();
    }

    // =========================================================================
    // JSON schema — batch build (unchanged from v1)
    // =========================================================================

    public class BuildRequest
    {
        [JsonProperty("job_id")]           public string JobId           { get; set; } = "";
        [JsonProperty("pdf_filename")]     public string PdfFilename     { get; set; } = "";
        [JsonProperty("transaction_json")] public string TransactionJson { get; set; } = "";
    }

    public class RevitTransaction
    {
        [JsonProperty("levels")]             public List<LevelItem>   Levels            { get; set; } = new List<LevelItem>();
        [JsonProperty("grids")]              public List<GridItem>    Grids             { get; set; } = new List<GridItem>();
        [JsonProperty("columns")]            public List<ColumnItem>  Columns           { get; set; } = new List<ColumnItem>();
        [JsonProperty("walls")]              public List<WallItem>    Walls             { get; set; } = new List<WallItem>();
        [JsonProperty("slabs")]              public List<SlabItem>    Slabs             { get; set; } = new List<SlabItem>();
        [JsonProperty("structural_framing")] public List<FramingItem> StructuralFraming { get; set; } = new List<FramingItem>();
    }

    public class FramingItem
    {
        [JsonProperty("id")]          public string    Id         { get; set; } = "";
        [JsonProperty("start_point")] public PointData StartPoint { get; set; } = new PointData();
        [JsonProperty("end_point")]   public PointData EndPoint   { get; set; } = new PointData();
        [JsonProperty("width")]       public double    Width      { get; set; } = 200;
        [JsonProperty("depth")]       public double    Depth      { get; set; } = 500;
        [JsonProperty("level")]       public string    Level      { get; set; } = "Level 0";
        [JsonProperty("family_type")] public string    FamilyType { get; set; } = "";
    }

    public class LevelItem
    {
        [JsonProperty("name")]      public string Name      { get; set; } = "";
        [JsonProperty("elevation")] public double Elevation { get; set; }
    }

    public class GridItem
    {
        [JsonProperty("name")]  public string    Name  { get; set; } = "";
        [JsonProperty("start")] public PointData Start { get; set; } = new PointData();
        [JsonProperty("end")]   public PointData End   { get; set; } = new PointData();
    }

    public class ColumnItem
    {
        [JsonProperty("id")]         public string    Id        { get; set; } = "";
        [JsonProperty("location")]   public PointData Location  { get; set; } = new PointData();
        [JsonProperty("width")]      public double    Width     { get; set; } = 300;
        [JsonProperty("depth")]      public double    Depth     { get; set; } = 300;
        [JsonProperty("height")]     public double    Height    { get; set; } = 2800;
        [JsonProperty("level")]      public string    Level     { get; set; } = "Level 0";
        [JsonProperty("top_level")]  public string    TopLevel  { get; set; } = "Level 1";
        [JsonProperty("shape")]      public string    Shape     { get; set; } = "";
        [JsonProperty("type_mark")]  public string    TypeMark  { get; set; } = "";
    }

    public class WallItem
    {
        [JsonProperty("id")]            public string    Id           { get; set; } = "";
        [JsonProperty("start_point")]   public PointData StartPoint   { get; set; } = new PointData();
        [JsonProperty("end_point")]     public PointData EndPoint     { get; set; } = new PointData();
        [JsonProperty("thickness")]     public double    Thickness    { get; set; } = 200;
        [JsonProperty("height")]        public double    Height       { get; set; } = 2800;
        [JsonProperty("level")]         public string    Level        { get; set; } = "Level 0";
        [JsonProperty("is_structural")] public bool      IsStructural { get; set; }
    }

    public class SlabItem
    {
        [JsonProperty("id")]              public string          Id             { get; set; } = "";
        [JsonProperty("boundary_points")] public List<PointData> BoundaryPoints { get; set; } = new List<PointData>();
        [JsonProperty("thickness")]       public double          Thickness      { get; set; } = 200;
        [JsonProperty("elevation")]       public double          Elevation      { get; set; } = 0;
        [JsonProperty("level")]           public string          Level          { get; set; } = "Level 0";
    }

    public class PointData
    {
        [JsonProperty("x")] public double X { get; set; }
        [JsonProperty("y")] public double Y { get; set; }
        [JsonProperty("z")] public double Z { get; set; }
    }
}