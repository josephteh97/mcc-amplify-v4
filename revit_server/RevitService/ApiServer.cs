using System;
using System.Net;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using Newtonsoft.Json;
using System.Collections.Generic;

namespace RevitService
{
    public class ApiServer
    {
        private HttpListener listener;
        private ModelBuilder modelBuilder;
        
        public ApiServer(ModelBuilder builder)
        {
            listener = new HttpListener();
            listener.Prefixes.Add("http://+:5000/");
            modelBuilder = builder;
        }
        
        public void Start()
        {
            listener.Start();
            Console.WriteLine("Revit API Server started on port 5000");
            
            Task.Run(() => Listen());
        }
        
        private async void Listen()
        {
            while (listener.IsListening)
            {
                var context = await listener.GetContextAsync();
                ProcessRequest(context);
            }
        }
        
        private async void ProcessRequest(HttpListenerContext context)
        {
            try
            {
                if (context.Request.HttpMethod == "POST" && context.Request.Url.AbsolutePath == "/build-model")
                {
                    await HandleBuildModel(context);
                }
                else if (context.Request.HttpMethod == "POST" && context.Request.Url.AbsolutePath == "/render-model")
                {
                    await HandleRenderModel(context);
                }
                else if (context.Request.HttpMethod == "GET" && context.Request.Url.AbsolutePath == "/health")
                {
                    byte[] response = Encoding.UTF8.GetBytes("Revit service healthy");
                    context.Response.ContentType = "text/plain";
                    context.Response.ContentLength64 = response.Length;
                    context.Response.OutputStream.Write(response, 0, response.Length);
                    context.Response.OutputStream.Close();
                }
                else
                {
                    context.Response.StatusCode = 404;
                    context.Response.OutputStream.Close();
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error: {ex.Message}");
                context.Response.StatusCode = 500;
                byte[] error = Encoding.UTF8.GetBytes(ex.Message);
                context.Response.OutputStream.Write(error, 0, error.Length);
                context.Response.OutputStream.Close();
            }
        }

        private async Task HandleBuildModel(HttpListenerContext context)
        {
            string requestBody;
            using (var reader = new StreamReader(context.Request.InputStream))
            {
                requestBody = await reader.ReadToEndAsync();
            }
            
            var request = JsonConvert.DeserializeObject<BuildRequest>(requestBody);
            if (request == null)
                throw new Exception("Invalid build request JSON.");

            string outputPath = Path.Combine(@"C:\RevitOutput", $"{request.JobId}.rvt");
            Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);

            // BuildModel takes the raw JSON string and deserialises it internally
            string resultPath = modelBuilder.BuildModel(request.TransactionJson, outputPath);
            
            byte[] rvtFile = File.ReadAllBytes(resultPath);
            context.Response.ContentType = "application/octet-stream";
            context.Response.ContentLength64 = rvtFile.Length;
            context.Response.AddHeader("Content-Disposition", $"attachment; filename={request.JobId}.rvt");
            context.Response.OutputStream.Write(rvtFile, 0, rvtFile.Length);
            context.Response.OutputStream.Close();
            
            Console.WriteLine($"Model built successfully: {request.JobId}");
        }

        private async Task HandleRenderModel(HttpListenerContext context)
        {
            // Simple Multipart Parser logic
            string contentType = context.Request.ContentType;
            if (string.IsNullOrEmpty(contentType) || !contentType.Contains("boundary="))
            {
                throw new Exception("Invalid Content-Type: Missing boundary");
            }
            
            string boundary = "--" + contentType.Split(new[] { "boundary=" }, StringSplitOptions.None)[1];
            
            // Read input stream
            byte[] buffer = new byte[context.Request.ContentLength64];
            await context.Request.InputStream.ReadAsync(buffer, 0, buffer.Length);
            
            // NOTE: This is a very simplified parser for demonstration.
            // In a production C# service, assume we are receiving ONE file field named 'file'.
            // We search for the file content between boundaries.
            
            // 1. Find start of file data
            // Look for: Content-Type: application/octet-stream (or similar) -> \r\n\r\n -> DATA
            
            string dataString = Encoding.GetEncoding("iso-8859-1").GetString(buffer);
            string fileHeader = "Content-Type: application/octet-stream";
            int headerIndex = dataString.IndexOf(fileHeader);
            
            if (headerIndex == -1) throw new Exception("Could not find file content in multipart request");
            
            int dataStartIndex = dataString.IndexOf("\r\n\r\n", headerIndex) + 4;
            
            // 2. Find end of file data (next boundary)
            int dataEndIndex = dataString.IndexOf(boundary, dataStartIndex) - 2; // -2 for \r\n before boundary
            
            if (dataStartIndex < 0 || dataEndIndex < dataStartIndex) throw new Exception("Failed to parse file boundaries");
            
            // 3. Extract file bytes
            int fileLength = dataEndIndex - dataStartIndex;
            byte[] fileBytes = new byte[fileLength];
            Array.Copy(buffer, dataStartIndex, fileBytes, 0, fileLength);

            // 4. Get Job ID (from header or new guid)
            string jobId = context.Request.Headers["X-Job-ID"] ?? Guid.NewGuid().ToString();
            
            string outputDir = Path.Combine(@"C:\RevitOutput", jobId);
            Directory.CreateDirectory(outputDir);
            
            string tempRvtPath = Path.Combine(outputDir, "input.rvt");
            File.WriteAllBytes(tempRvtPath, fileBytes);
            
            Console.WriteLine($"Received RVT file for Job {jobId}, size: {fileLength} bytes");
            
            // 5. Render
            string renderPath = modelBuilder.RenderModel(tempRvtPath, outputDir);
            
            byte[] imgFile = File.ReadAllBytes(renderPath);
            context.Response.ContentType = "image/png";
            context.Response.ContentLength64 = imgFile.Length;
            context.Response.OutputStream.Write(imgFile, 0, imgFile.Length);
            context.Response.OutputStream.Close();
        }
        
        public void Stop()
        {
            listener.Stop();
            listener.Close();
        }
    }

    // Request model sent by the Python revit_client.build_model()
    public class BuildRequest
    {
        [JsonProperty("job_id")]
        public string JobId { get; set; } = string.Empty;

        /// <summary>
        /// The RevitTransaction JSON serialised as a string.
        /// ModelBuilder.BuildModel() deserialises it internally.
        /// </summary>
        [JsonProperty("transaction_json")]
        public string TransactionJson { get; set; } = string.Empty;
    }
}
