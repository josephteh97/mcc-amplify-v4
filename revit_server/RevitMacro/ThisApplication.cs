// =============================================================================
// Revit 2023 Application Macro — Amplify AI BIM Builder
// =============================================================================
//
// HOW TO USE (Macro Manager approach)
// ------------------------------------
// 1. In Revit 2023: Manage tab → Macros → Macro Manager
// 2. Click "Application" tab → "Edit" to open the SharpDevelop IDE
//    (or go to Manage → Macros → Edit to open the IDE)
// 3. Replace the contents of "ThisApplication.cs" with this entire file.
// 4. Add reference to Newtonsoft.Json.dll (find it in Revit's install folder,
//    e.g. C:\Program Files\Autodesk\Revit 2023\Newtonsoft.Json.dll)
//    → Right-click References in IDE → Add Reference → Browse → select DLL
// 5. Click Build. If it compiles, close the IDE.
// 6. In Macro Manager, select "BuildRevitModelFromJSON" and click Run.
//
// WORKFLOW
// ---------
// The Python backend writes the transaction JSON to:
//   C:\RevitOutput\pending.json
//
// This macro reads that file, builds the BIM model in a NEW project
// created from the Structural template, and saves the output to:
//   C:\RevitOutput\{job_id}.rvt
//
// After saving, it writes a completion marker:
//   C:\RevitOutput\{job_id}.done
//
// The Python revit_client.py can be updated to poll for the .done file
// and then serve the .rvt back to the user.
//
// INTEGRATION WITH HTTP PIPELINE (optional)
// ------------------------------------------
// If you want the macro to run automatically triggered by the Python pipeline:
// 1. Have the Python server write JSON to C:\RevitOutput\pending.json
// 2. The Python server also writes C:\RevitOutput\trigger.txt  with the job_id
// 3. Run this macro manually OR set up the Startup handler to poll every 5 s
//    using a DispatcherTimer (see commented code in OnStartup below).
// =============================================================================

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Autodesk.Revit.ApplicationServices;
using Autodesk.Revit.DB;
using Autodesk.Revit.DB.Structure;
using Autodesk.Revit.UI;
using Newtonsoft.Json;

namespace RevitMacros
{
    [Autodesk.Revit.Attributes.Transaction(Autodesk.Revit.Attributes.TransactionMode.Manual)]
    [Autodesk.Revit.Attributes.Regeneration(Autodesk.Revit.Attributes.RegenerationOption.Manual)]
    public class ThisApplication
    {
        // Injected by the Revit macro framework at runtime
        private UIApplication m_uiApp;

        // ── Macro entry points (appear in Macro Manager) ──────────────────────

        /// <summary>
        /// Build a Revit model from C:\RevitOutput\pending.json
        /// and save it to C:\RevitOutput\{job_id}.rvt
        /// </summary>
        public void BuildRevitModelFromJSON()
        {
            const string PendingPath = @"C:\RevitOutput\pending.json";
            const string OutputDir   = @"C:\RevitOutput";

            if (!File.Exists(PendingPath))
            {
                TaskDialog.Show("Amplify BIM Builder",
                    $"No pending job found.\nExpected: {PendingPath}");
                return;
            }

            try
            {
                string json = File.ReadAllText(PendingPath);
                var recipe = JsonConvert.DeserializeObject<RevitTransaction>(
                    json,
                    new JsonSerializerSettings
                    {
                        MissingMemberHandling = MissingMemberHandling.Ignore,
                        NullValueHandling     = NullValueHandling.Ignore,
                    }) ?? throw new Exception("Could not parse pending.json");

                string jobId      = recipe.JobId ?? Guid.NewGuid().ToString();
                string outputPath = Path.Combine(OutputDir, $"{jobId}.rvt");

                Directory.CreateDirectory(OutputDir);

                Application app = m_uiApp.Application;

                // Build the model
                var builder = new BimModelBuilder(app);
                builder.Build(recipe, outputPath);

                // Write completion marker so the Python side knows it's done
                File.WriteAllText(Path.Combine(OutputDir, $"{jobId}.done"), outputPath);

                // Delete the pending file so it isn't reprocessed
                File.Delete(PendingPath);

                TaskDialog.Show("Amplify BIM Builder",
                    $"Model built successfully!\n\nSaved to:\n{outputPath}");
            }
            catch (Exception ex)
            {
                TaskDialog.Show("Amplify BIM Builder — Error", ex.ToString());
            }
        }

        // ── Macro framework callbacks (required but unused here) ──────────────

        internal void OnStartup(UIControlledApplication app)  { /* optional: start polling timer here */ }
        internal void OnShutdown(UIControlledApplication app) { }
    }

    // =========================================================================
    // BIM model builder
    // =========================================================================

    public class BimModelBuilder
    {
        private readonly Application _app;
        private const double MM = 1.0 / 304.8;   // mm → internal Revit feet

        private const string STRUCTURAL_TEMPLATE =
            @"C:\ProgramData\Autodesk\RVT 2023\Templates\Structural Analysis-Default.rte";
        private const string ARCH_TEMPLATE =
            @"C:\ProgramData\Autodesk\RVT 2023\Templates\Architectural Template.rte";

        public BimModelBuilder(Application app) { _app = app; }

        public void Build(RevitTransaction recipe, string outputPath)
        {
            string template = File.Exists(STRUCTURAL_TEMPLATE)
                              ? STRUCTURAL_TEMPLATE
                              : ARCH_TEMPLATE;

            Document doc = _app.NewProjectDocument(template);

            try
            {
                using (var t = new Transaction(doc, "Amplify AI — Build BIM from Floor Plan"))
                {
                    t.Start();
                    CreateLevels (doc, recipe.Levels);
                    CreateGrids  (doc, recipe.Grids);
                    CreateColumns(doc, recipe.Columns);
                    CreateWalls  (doc, recipe.Walls);
                    CreateFloors (doc, recipe.Floors);
                    t.Commit();
                }

                doc.SaveAs(outputPath, new SaveAsOptions { OverwriteExistingFile = true });
            }
            finally
            {
                doc.Close(false);
            }
        }

        // ── Levels ────────────────────────────────────────────────────────────

        private void CreateLevels(Document doc, List<LevelItem> levels)
        {
            var effective = (levels != null && levels.Count >= 2)
                ? levels
                : new List<LevelItem>
                  {
                      new LevelItem { Name = "Level 0", Elevation = 0 },
                      new LevelItem { Name = "Level 1", Elevation = 3000 },
                  };

            foreach (var lv in effective)
            {
                bool exists = new FilteredElementCollector(doc)
                    .OfClass(typeof(Level)).Cast<Level>()
                    .Any(x => x.Name.Equals(lv.Name, StringComparison.OrdinalIgnoreCase));

                if (exists) continue;

                Level level = Level.Create(doc, lv.Elevation * MM);
                level.Name = lv.Name;
            }
        }

        // ── Structural grid lines ─────────────────────────────────────────────

        private void CreateGrids(Document doc, List<GridItem> grids)
        {
            if (grids == null) return;

            foreach (var g in grids)
            {
                if (g.Start == null || g.End == null) continue;

                XYZ start = Pt(g.Start), end = Pt(g.End);
                if (start.DistanceTo(end) < 1e-9) continue;

                Grid grid = Grid.Create(doc, Line.CreateBound(start, end));
                if (!string.IsNullOrWhiteSpace(g.Name))
                    try { grid.Name = g.Name; } catch { /* duplicate name */ }
            }
        }

        // ── Structural columns ────────────────────────────────────────────────

        private void CreateColumns(Document doc, List<ColumnItem> columns)
        {
            if (columns == null || columns.Count == 0) return;

            FamilySymbol baseSymbol = FindOrLoadColumnFamily(doc);
            if (baseSymbol == null)
            {
                // No column family available — skip silently
                return;
            }

            if (!baseSymbol.IsActive) baseSymbol.Activate();

            foreach (var col in columns)
            {
                if (col.Location == null) continue;

                Level baseLevel = GetLevel(doc, col.Level ?? "Level 0");
                if (baseLevel == null) continue;

                // Match column size to a type (duplicate if needed)
                FamilySymbol sym = GetOrDuplicateColumnType(
                    doc, baseSymbol, col.Width, col.Depth) ?? baseSymbol;

                if (!sym.IsActive) sym.Activate();

                FamilyInstance inst = doc.Create.NewFamilyInstance(
                    Pt(col.Location), sym, baseLevel, StructuralType.Column);

                Level topLevel = GetLevel(doc, col.TopLevel ?? "Level 1");
                if (topLevel != null)
                    inst.get_Parameter(BuiltInParameter.FAMILY_TOP_LEVEL_PARAM)
                        ?.Set(topLevel.Id);
            }
        }

        private FamilySymbol FindOrLoadColumnFamily(Document doc)
        {
            // 1. Find any structural column symbol already loaded in the project
            var existing = new FilteredElementCollector(doc)
                .OfClass(typeof(FamilySymbol)).Cast<FamilySymbol>()
                .FirstOrDefault(s => s.Category?.Id?.IntegerValue ==
                                     (int)BuiltInCategory.OST_StructuralColumns);
            if (existing != null) return existing;

            // 2. Search Revit 2023 library root recursively — handles any locale/drive
            var searchRoots = new[]
            {
                Path.Combine(
                    Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData),
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

                foreach (var path in rfas.OrderBy(p =>
                    Path.GetFileName(p).StartsWith("M_", StringComparison.OrdinalIgnoreCase) ? 0 : 1))
                {
                    try
                    {
                        if (doc.LoadFamily(path, out Family fam) && fam != null)
                        {
                            var sym = fam.GetFamilySymbolIds()
                                         .Select(id => doc.GetElement(id) as FamilySymbol)
                                         .FirstOrDefault(s => s != null);
                            if (sym != null) return sym;
                        }
                    }
                    catch { }
                }
            }

            return null;
        }

        private FamilySymbol GetOrDuplicateColumnType(
            Document doc, FamilySymbol baseSymbol, double wMm, double dMm)
        {
            double w = Math.Round(wMm / 50.0) * 50;
            double d = Math.Round(dMm / 50.0) * 50;
            string name = $"{(int)w}x{(int)d}mm";

            var found = new FilteredElementCollector(doc)
                .OfClass(typeof(FamilySymbol)).Cast<FamilySymbol>()
                .FirstOrDefault(s => s.FamilyName == baseSymbol.FamilyName &&
                                     s.Name.Equals(name, StringComparison.OrdinalIgnoreCase));
            if (found != null) return found;

            try
            {
                var nt = baseSymbol.Duplicate(name) as FamilySymbol;
                if (nt == null) return null;

                SetMm(nt, "b",     w);
                SetMm(nt, "h",     d);
                SetMm(nt, "Width", w);
                SetMm(nt, "Depth", d);
                return nt;
            }
            catch { return null; }
        }

        private static void SetMm(FamilySymbol sym, string pName, double mm)
        {
            var p = sym.LookupParameter(pName);
            if (p != null && !p.IsReadOnly && p.StorageType == StorageType.Double)
                p.Set(mm / 304.8);
        }

        // ── Walls ─────────────────────────────────────────────────────────────

        private void CreateWalls(Document doc, List<WallItem> walls)
        {
            if (walls == null || walls.Count == 0) return;

            WallType wt = new FilteredElementCollector(doc)
                .OfClass(typeof(WallType)).Cast<WallType>().FirstOrDefault();
            if (wt == null) return;

            foreach (var w in walls)
            {
                if (w.StartPoint == null || w.EndPoint == null) continue;

                XYZ s = Pt(w.StartPoint), e = Pt(w.EndPoint);
                if (s.DistanceTo(e) < 1e-9) continue;

                Level lv = GetLevel(doc, w.Level ?? "Level 0");
                if (lv == null) continue;

                Wall.Create(doc, Line.CreateBound(s, e), wt.Id, lv.Id,
                            (w.Height > 0 ? w.Height : 2800) * MM,
                            0.0, false, w.IsStructural);
            }
        }

        // ── Floors ────────────────────────────────────────────────────────────

        private void CreateFloors(Document doc, List<FloorItem> floors)
        {
            if (floors == null || floors.Count == 0) return;

            FloorType ft = new FilteredElementCollector(doc)
                .OfClass(typeof(FloorType)).Cast<FloorType>().FirstOrDefault();
            if (ft == null) return;

            foreach (var fl in floors)
            {
                if (fl.BoundaryPoints == null || fl.BoundaryPoints.Count < 3) continue;

                Level lv = GetLevel(doc, fl.Level ?? "Level 0");
                if (lv == null) continue;

                var loop = new CurveLoop();
                int n = fl.BoundaryPoints.Count;
                for (int i = 0; i < n; i++)
                {
                    var p1 = fl.BoundaryPoints[i];
                    var p2 = fl.BoundaryPoints[(i + 1) % n];
                    loop.Append(Line.CreateBound(
                        new XYZ(p1.X * MM, p1.Y * MM, 0),
                        new XYZ(p2.X * MM, p2.Y * MM, 0)));
                }

                Floor.Create(doc, new List<CurveLoop> { loop }, ft.Id, lv.Id);
            }
        }

        // ── Helpers ───────────────────────────────────────────────────────────

        private XYZ Pt(PointData p) =>
            new XYZ(p.X * MM, p.Y * MM, p.Z * MM);

        private Level GetLevel(Document doc, string name) =>
            new FilteredElementCollector(doc).OfClass(typeof(Level)).Cast<Level>()
                .FirstOrDefault(l => l.Name.Equals(name, StringComparison.OrdinalIgnoreCase));
    }

    // =========================================================================
    // JSON schema — matches Python geometry_generator.py flat output
    // =========================================================================

    public class RevitTransaction
    {
        [JsonProperty("job_id")]   public string         JobId   { get; set; }
        [JsonProperty("levels")]   public List<LevelItem>  Levels  { get; set; } = new();
        [JsonProperty("grids")]    public List<GridItem>   Grids   { get; set; } = new();
        [JsonProperty("columns")]  public List<ColumnItem> Columns { get; set; } = new();
        [JsonProperty("walls")]    public List<WallItem>   Walls   { get; set; } = new();
        [JsonProperty("floors")]   public List<FloorItem>  Floors  { get; set; } = new();
    }

    public class LevelItem
    {
        [JsonProperty("name")]      public string Name      { get; set; } = "";
        [JsonProperty("elevation")] public double Elevation { get; set; }
    }

    public class GridItem
    {
        [JsonProperty("name")]  public string    Name  { get; set; } = "";
        [JsonProperty("start")] public PointData Start { get; set; } = new();
        [JsonProperty("end")]   public PointData End   { get; set; } = new();
    }

    public class ColumnItem
    {
        [JsonProperty("id")]        public string    Id       { get; set; } = "";
        [JsonProperty("location")]  public PointData Location { get; set; } = new();
        [JsonProperty("width")]     public double    Width    { get; set; } = 300;
        [JsonProperty("depth")]     public double    Depth    { get; set; } = 300;
        [JsonProperty("height")]    public double    Height   { get; set; } = 2800;
        [JsonProperty("level")]     public string    Level    { get; set; } = "Level 0";
        [JsonProperty("top_level")] public string    TopLevel { get; set; } = "Level 1";
    }

    public class WallItem
    {
        [JsonProperty("id")]            public string    Id           { get; set; } = "";
        [JsonProperty("start_point")]   public PointData StartPoint   { get; set; } = new();
        [JsonProperty("end_point")]     public PointData EndPoint     { get; set; } = new();
        [JsonProperty("thickness")]     public double    Thickness    { get; set; } = 200;
        [JsonProperty("height")]        public double    Height       { get; set; } = 2800;
        [JsonProperty("level")]         public string    Level        { get; set; } = "Level 0";
        [JsonProperty("is_structural")] public bool      IsStructural { get; set; }
    }

    public class FloorItem
    {
        [JsonProperty("id")]              public string          Id             { get; set; } = "";
        [JsonProperty("boundary_points")] public List<PointData> BoundaryPoints { get; set; } = new();
        [JsonProperty("level")]           public string          Level          { get; set; } = "Level 0";
        [JsonProperty("is_structural")]   public bool            IsStructural   { get; set; } = true;
    }

    public class PointData
    {
        [JsonProperty("x")] public double X { get; set; }
        [JsonProperty("y")] public double Y { get; set; }
        [JsonProperty("z")] public double Z { get; set; }
    }
}
