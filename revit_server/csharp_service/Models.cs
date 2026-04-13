using System.Collections.Generic;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace RevitService
{
    // ========================================================================
    // Command Pattern Architecture
    // ========================================================================

    public interface IRevitCommand
    {
        string Type { get; }
        void Execute(ModelBuilder builder);
    }

    public class RevitRecipe
    {
        public string Version { get; set; } = "1.0";
        public ProjectInfo ProjectInfo { get; set; } = new();
        public List<RecipeStep> Steps { get; set; } = new();
    }

    public class RecipeStep
    {
        public string CommandType { get; set; } = ""; // e.g., "Wall.Create", "Door.Create"
        public JObject Parameters { get; set; } = new(); // Dynamic parameters
    }

    // ========================================================================
    // Concrete Command Data Models (for Deserialization)
    // ========================================================================

    public class WallCreateData
    {
        public Point3D Start { get; set; }
        public Point3D End { get; set; }
        public string WallType { get; set; } = "Generic - 200mm";
        public string Level { get; set; } = "Level 1";
        public double Height { get; set; } = 3000;
        public bool Structural { get; set; } = false;
    }

    public class DoorCreateData
    {
        public Point3D Location { get; set; }
        public string Family { get; set; } = "M_Single-Flush";
        public string Symbol { get; set; } = "0915 x 2134mm";
        public string HostWallId { get; set; } // Can be index or ID
        public string Level { get; set; } = "Level 1";
    }
    
    public class WindowCreateData
    {
        public Point3D Location { get; set; }
        public string Family { get; set; } = "M_Fixed";
        public string Symbol { get; set; } = "0406 x 0610mm";
        public string HostWallId { get; set; }
        public string Level { get; set; } = "Level 1";
    }
    
    public class ColumnCreateData
    {
        public Point3D Location { get; set; }
        public string Family { get; set; } = "M_Rectangular Column";
        public string Symbol { get; set; } = "457 x 457mm";
        public string Level { get; set; } = "Level 1";
        public double Height { get; set; } = 3000;
    }

    public class FloorCreateData
    {
        public List<Point3D> Boundary { get; set; } = new();
        public string FloorType { get; set; } = "Generic - 150mm";
        public string Level { get; set; } = "Level 1";
    }

    public class RenderModelData
    {
        public string ViewName { get; set; } = "3D View";
        public string OutputFormat { get; set; } = "png"; // png, jpg, gltf
        public int Width { get; set; } = 1920;
        public int Height { get; set; } = 1080;
    }

    // ========================================================================
    // Existing Models (Kept for compatibility or reused)
    // ========================================================================

    public class Config
    {
        public RevitSettings RevitSettings { get; set; } = new();
        public ApiSettings ApiSettings { get; set; } = new();
        public LoggingSettings LoggingSettings { get; set; } = new();
    }

    public class RevitSettings
    {
        public string Version { get; set; } = "2023";
        public string TemplatePath { get; set; } = @"C:\ProgramData\Autodesk\RVT 2023\Templates\Architectural Template.rte";
        public string OutputDirectory { get; set; } = @"C:\RevitOutput";
        public bool EnableHeadless { get; set; } = true;
    }

    public class ApiSettings
    {
        public string Host { get; set; } = "0.0.0.0";
        public int Port { get; set; } = 5000;
        public string ApiKey { get; set; } = "change-this-key";
        public int TimeoutSeconds { get; set; } = 300;
    }

    public class LoggingSettings
    {
        public string Level { get; set; } = "Information";
        public string Directory { get; set; } = "logs";
    }

    public class BuildRequest
    {
        public string JobId { get; set; } = "";
        public string TransactionJson { get; set; } = "";
    }
    
    public class RenderRequest
    {
        public string JobId { get; set; }
        public string RvtFilePath { get; set; }
    }

    public class ProjectInfo
    {
        public string Name { get; set; } = "";
        public string Author { get; set; } = "";
        public string CreatedDate { get; set; } = "";
    }

    public class Point3D
    {
        public double X { get; set; }
        public double Y { get; set; }
        public double Z { get; set; }
    }
}
