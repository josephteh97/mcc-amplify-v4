using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Autodesk.Revit.ApplicationServices;
using Autodesk.Revit.DB;
using Autodesk.Revit.DB.Architecture;
using Autodesk.Revit.DB.Structure;
using Serilog;
using Newtonsoft.Json.Linq;

namespace RevitService
{
    public class ModelBuilder
    {
        private readonly Application _app;
        private readonly Config _config;
        private Document _doc;
        private readonly CommandProcessor _commandProcessor;
        private const double MM_TO_FEET = 1.0 / 304.8;

        public ModelBuilder(Application app, Config config)
        {
            _app = app;
            _config = config;
            _commandProcessor = new CommandProcessor();
        }

        public Document Doc => _doc;

        public async Task<string> BuildModel(RevitRecipe recipe, string outputPath)
        {
            return await Task.Run(() =>
            {
                try
                {
                    Log.Information("Creating new Revit document...");
                    _doc = _app.NewProjectDocument(_config.RevitSettings.TemplatePath);
                    
                    using (Transaction trans = new Transaction(_doc, "Build Model via Recipe"))
                    {
                        trans.Start();
                        try
                        {
                            foreach (var step in recipe.Steps)
                            {
                                Log.Information($"Executing command: {step.CommandType}");
                                IRevitCommand command = _commandProcessor.CreateCommand(step.CommandType, step.Parameters);
                                command?.Execute(this);
                            }

                            trans.Commit();
                            Log.Information("✓ Transaction committed successfully");
                        }
                        catch (Exception ex)
                        {
                            trans.RollBack();
                            Log.Error(ex, "Transaction failed, rolling back");
                            throw;
                        }
                    }

                    Log.Information($"Saving document to: {outputPath}");
                    SaveAsOptions saveOptions = new SaveAsOptions { OverwriteExistingFile = true };
                    _doc.SaveAs(outputPath, saveOptions);
                    _doc.Close(false);
                    _doc = null;

                    Log.Information("✓ Model saved and closed");
                    return outputPath;
                }
                catch (Exception ex)
                {
                    Log.Error(ex, "Failed to build model");
                    if (_doc != null)
                    {
                        _doc.Close(false);
                        _doc = null;
                    }
                    throw;
                }
            });
        }
        
        public string RenderModel(string rvtPath, string outputDir)
        {
             try
             {
                 Log.Information($"Opening model for rendering: {rvtPath}");
                 _doc = _app.OpenDocumentFile(rvtPath);
                 
                 // Find 3D View
                 View3D view3d = new FilteredElementCollector(_doc)
                     .OfClass(typeof(View3D))
                     .Cast<View3D>()
                     .FirstOrDefault(v => v.Name == "{3D}" || v.ViewType == ViewType.ThreeD);

                 if (view3d == null)
                     throw new Exception("No 3D view found in model");

                 string outputPath = System.IO.Path.Combine(outputDir, "render.png");
                 
                 ImageExportOptions options = new ImageExportOptions
                 {
                     ZoomType = ZoomFitType.FitToPage,
                     PixelSize = 1920,
                     FilePath = outputPath,
                     FitDirection = FitDirectionType.Horizontal,
                     HLRandWFViewsFileType = ImageFileType.PNG,
                     ShadowViewsFileType = ImageFileType.PNG,
                     ImageResolution = ImageResolution.DPI_300,
                     ExportRange = ExportRange.SetOfViews,
                 };
                 options.SetViewsAndSheets(new List<ElementId> { view3d.Id });

                 _doc.ExportImage(options);
                 _doc.Close(false);
                 _doc = null;
                 
                 Log.Information($"Render saved to: {outputPath}");
                 return outputPath;
             }
             catch (Exception ex)
             {
                 Log.Error(ex, "Failed to render model");
                 if (_doc != null) { _doc.Close(false); _doc = null; }
                 throw;
             }
        }

        // ========================================================================
        // Helper Methods for Commands
        // ========================================================================

        public void CreateWall(WallCreateData data)
        {
            XYZ start = new XYZ(data.Start.X * MM_TO_FEET, data.Start.Y * MM_TO_FEET, 0);
            XYZ end = new XYZ(data.End.X * MM_TO_FEET, data.End.Y * MM_TO_FEET, 0);
            Line line = Line.CreateBound(start, end);

            WallType wallType = GetElementByName<WallType>(data.WallType);
            Level level = GetElementByName<Level>(data.Level);

            if (wallType != null && level != null)
            {
                Wall.Create(_doc, line, wallType.Id, level.Id, data.Height * MM_TO_FEET, 0, false, data.Structural);
            }
        }

        public void CreateDoor(DoorCreateData data)
        {
            FamilySymbol symbol = GetFamilySymbol(data.Family, data.Symbol);
            if (symbol == null) return;
            if (!symbol.IsActive) symbol.Activate();

            Wall hostWall = _doc.GetElement(new ElementId(int.Parse(data.HostWallId))) as Wall;
            Level level = GetElementByName<Level>(data.Level);
            XYZ location = new XYZ(data.Location.X * MM_TO_FEET, data.Location.Y * MM_TO_FEET, 0);

            if (hostWall != null && level != null)
            {
                _doc.Create.NewFamilyInstance(location, symbol, hostWall, level, StructuralType.NonStructural);
            }
        }

        public void CreateColumn(ColumnCreateData data)
        {
             FamilySymbol symbol = GetFamilySymbol(data.Family, data.Symbol);
             if (symbol == null) return;
             if (!symbol.IsActive) symbol.Activate();
             
             Level level = GetElementByName<Level>(data.Level);
             XYZ location = new XYZ(data.Location.X * MM_TO_FEET, data.Location.Y * MM_TO_FEET, 0);
             
             if (level != null)
             {
                 _doc.Create.NewFamilyInstance(location, symbol, level, StructuralType.Column);
             }
        }

        private T GetElementByName<T>(string name) where T : Element
        {
            return new FilteredElementCollector(_doc)
                .OfClass(typeof(T))
                .Cast<T>()
                .FirstOrDefault(e => e.Name == name);
        }

        private FamilySymbol GetFamilySymbol(string familyName, string symbolName)
        {
            return new FilteredElementCollector(_doc)
                .OfClass(typeof(FamilySymbol))
                .Cast<FamilySymbol>()
                .FirstOrDefault(x => x.FamilyName == familyName && x.Name == symbolName);
        }
    }

    // ========================================================================
    // Command Processor & Concrete Commands
    // ========================================================================

    public class CommandProcessor
    {
        public IRevitCommand CreateCommand(string type, JObject parameters)
        {
            return type switch
            {
                "Wall.Create" => new CreateWallCommand(parameters.ToObject<WallCreateData>()),
                "Door.Create" => new CreateDoorCommand(parameters.ToObject<DoorCreateData>()),
                "Column.Create" => new CreateColumnCommand(parameters.ToObject<ColumnCreateData>()),
                _ => null
            };
        }
    }

    public class CreateWallCommand : IRevitCommand
    {
        private readonly WallCreateData _data;
        public string Type => "Wall.Create";
        public CreateWallCommand(WallCreateData data) { _data = data; }
        public void Execute(ModelBuilder builder) { builder.CreateWall(_data); }
    }

    public class CreateDoorCommand : IRevitCommand
    {
        private readonly DoorCreateData _data;
        public string Type => "Door.Create";
        public CreateDoorCommand(DoorCreateData data) { _data = data; }
        public void Execute(ModelBuilder builder) { builder.CreateDoor(_data); }
    }
    
    public class CreateColumnCommand : IRevitCommand
    {
        private readonly ColumnCreateData _data;
        public string Type => "Column.Create";
        public CreateColumnCommand(ColumnCreateData data) { _data = data; }
        public void Execute(ModelBuilder builder) { builder.CreateColumn(_data); }
    }
}
