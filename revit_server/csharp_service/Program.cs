using System;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Linq;
using System.Threading.Tasks;
using Newtonsoft.Json;
using Serilog;
using Autodesk.Revit.UI;
using Autodesk.Revit.DB;
using Autodesk.Revit.ApplicationServices;

namespace RevitService
{
    public class Program : IExternalApplication
    {
        private static Config _config;
        private static bool _isRunning = true;
        private static RevitBuildHandler _handler;
        private static ExternalEvent _externalEvent;

        public Result OnStartup(UIControlledApplication application)
        {
            SetupLogging();
            Log.Information("Revit Socket Service - Starting inside Revit");

            _handler = new RevitBuildHandler();
            _externalEvent = ExternalEvent.Create(_handler);
            _config = LoadConfiguration();

            Task.Run(() => StartHttpServer());
            return Result.Succeeded;
        }

        public Result OnShutdown(UIControlledApplication application)
        {
            _isRunning = false;
            Log.Information("Shutting down Revit API Service");
            Log.CloseAndFlush();
            return Result.Succeeded;
        }

        static async Task Main(string[] args)
        {
            SetupLogging();
            Log.Information("Starting in Standalone Mode");
            TryInitStandalone();
            await StartHttpServer();
        }

        private static void SetupLogging()
        {
            if (Log.Logger.GetType().Name == "SilentLogger")
            {
                Log.Logger = new LoggerConfiguration()
                    .WriteTo.Console()
                    .WriteTo.File("logs/revit-service.log", rollingInterval: RollingInterval.Day)
                    .CreateLogger();
            }
        }

        static void TryInitStandalone()
        {
            try {
                _handler = new RevitBuildHandler();
                _externalEvent = ExternalEvent.Create(_handler);
            }
            catch {
                Log.Warning("⚠️ Revit not detected. ExternalEvents disabled.");
            }
        }

        static Config LoadConfiguration()
        {
            string configPath = "config.json";
            if (!File.Exists(configPath)) return new Config();
            string json = File.ReadAllText(configPath);
            return JsonConvert.DeserializeObject<Config>(json) ?? new Config();
        }

        static async Task StartHttpServer()
        {
            Socket listener = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
            try {
                listener.Bind(new IPEndPoint(IPAddress.Any, 49152));
                listener.Listen(100);
                Log.Information("🚀 SOCKET SERVER ACTIVE on port 49152");

                while (_isRunning) {
                    Socket handler = await listener.AcceptAsync();
                    _ = Task.Run(() => ProcessClient(handler));
                }
            }
            catch (Exception ex) { Log.Fatal(ex, "Socket failed"); }
        }

        private static void ProcessClient(Socket handler)
        {
            try {
                byte[] buffer = new byte[8192];
                int received = handler.Receive(buffer);
                if (received == 0) return;

                string request = Encoding.UTF8.GetString(buffer, 0, received);
                
                // Parse Request Path safely
                string[] lines = request.Split('\n');
                string requestPath = lines.Length > 0 && lines[0].Contains(" ") 
                    ? lines[0].Split(' ')[1].Trim() 
                    : "/";

                // Parse Request Body
                string requestBody = "";
                int bodyStart = request.IndexOf("\r\n\r\n");
                if (bodyStart != -1) requestBody = request.Substring(bodyStart + 4).Trim();

                string jsonResponse;
                if (requestPath == "/build") {
                    if (_handler != null && _externalEvent != null) {
                        _handler.Data = requestBody;
                        _externalEvent.Raise();
                        jsonResponse = "{\"status\":\"QUEUED\"}";
                    } else {
                        jsonResponse = "{\"status\":\"ERROR\", \"message\":\"Revit not linked\"}";
                    }
                } else {
                    jsonResponse = "{\"status\":\"OK\"}";
                }

                string response = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n" + jsonResponse;
                handler.Send(Encoding.UTF8.GetBytes(response));
            }
            catch (Exception ex) { Log.Error(ex, "ProcessClient Error"); }
            finally { handler.Close(); }
        }
    }

    public class RevitBuildHandler : IExternalEventHandler
    {
        public string Data { get; set; } = string.Empty;

        public void Execute(UIApplication app)
        {
            Document doc = app.ActiveUIDocument?.Document;
            if (doc == null) return;

            using (Transaction tx = new Transaction(doc, "Build from Ubuntu"))
            {
                tx.Start();
                Log.Information($"Revit logic executing with data: {Data}");
                // YOUR LOGIC HERE
                tx.Commit();
            }
        }
        public string GetName() => "Revit Build External Event";
    }
}