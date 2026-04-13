// windows_server/RevitService/ModelBuilder.cs
//
// Revit 2023 BIM Model Builder
// ============================
// Follows Autodesk Revit 2023 API conventions:
//   • All lengths are converted from millimetres (Python side) to internal
//     Revit units (decimal feet) using the constant MM_TO_FEET = 1/304.8.
//   • Grid lines are created with Grid.Create(Document, Line) and named
//     per the structural grid detected from the floor plan.
//   • Levels:  Level 0 (Ground Floor, 0 mm) and Level 1 (First Floor)
//     are always present — defaulting to a 3 000 mm storey height when
//     the number of storeys is unknown.
//   • Columns reference their top level by name from the transaction data
//     (never hard-coded).
//   • The transaction JSON schema is defined at the bottom of this file.

using Autodesk.Revit.ApplicationServices;
using Autodesk.Revit.DB;
using Autodesk.Revit.DB.Architecture;
using Autodesk.Revit.DB.Structure;
using System;
using System.Collections.Generic;
using System.Linq;
using Newtonsoft.Json;

namespace RevitService
{
    public class ModelBuilder
    {
        private Application revitApp;
        private Document doc;

        // Revit 2023 internal unit: decimal feet.
        // 1 mm = 1/304.8 ft  ≈  0.003280840 ft
        private const double MM_TO_FEET = 1.0 / 304.8;

        // Template shipped with Revit 2023
        private const string TEMPLATE_PATH =
            @"C:\ProgramData\Autodesk\RVT 2023\Templates\Architectural Template.rte";

        // Default storey height (mm) used when transaction carries no levels
        private const double DEFAULT_STOREY_HEIGHT_MM = 3000.0;

        public ModelBuilder()
        {
            revitApp = new Application();
        }

        // ------------------------------------------------------------------
        // Entry point
        // ------------------------------------------------------------------

        public string BuildModel(string transactionJson, string outputPath)
        {
            try
            {
                var transaction = JsonConvert.DeserializeObject<RevitTransaction>(
                    transactionJson,
                    new JsonSerializerSettings { MissingMemberHandling = MissingMemberHandling.Ignore }
                );
                if (transaction == null)
                    throw new ArgumentNullException(nameof(transactionJson),
                        "Failed to deserialise transaction JSON.");

                doc = revitApp.NewProjectDocument(TEMPLATE_PATH);

                using (Transaction trans = new Transaction(doc, "Build Floor Plan Model"))
                {
                    trans.Start();

                    // ── 1. Levels  (must be first — everything else references them)
                    CreateLevels(transaction.Levels);

                    // ── 2. Structural grid lines
                    CreateGrids(transaction.Grids);

                    // ── 3. Structural / architectural elements
                    CreateWalls(transaction.Walls);
                    CreateColumns(transaction.Columns);
                    CreateFloors(transaction.Floors);

                    // ── 4. Non-structural hosted elements
                    CreateDoors(transaction.Doors);
                    CreateWindows(transaction.Windows);

                    // ── 5. Rooms & views
                    CreateRooms(transaction.Rooms);
                    CreateViews(transaction.Views);

                    trans.Commit();
                }

                SaveAsOptions saveOptions = new SaveAsOptions { OverwriteExistingFile = true };
                doc.SaveAs(outputPath, saveOptions);
                doc.Close(false);

                return outputPath;
            }
            catch (Exception ex)
            {
                throw new Exception($"Revit model building failed: {ex.Message}", ex);
            }
        }

        // ------------------------------------------------------------------
        // 1. Levels
        // ------------------------------------------------------------------

        private void CreateLevels(List<LevelCommand>? levels)
        {
            // Ensure at minimum Level 0 (Ground Floor) and Level 1 (First Floor)
            // are always present, regardless of what the transaction provides.
            var effectiveLevels = (levels != null && levels.Count > 0)
                ? levels
                : new List<LevelCommand>
                {
                    new LevelCommand { Name = "Level 0", Elevation = 0.0 },
                    new LevelCommand { Name = "Level 1", Elevation = DEFAULT_STOREY_HEIGHT_MM },
                };

            // Guarantee Level 0 and Level 1 are always in the list
            if (!effectiveLevels.Any(l => l.Name == "Level 0"))
                effectiveLevels.Insert(0, new LevelCommand { Name = "Level 0", Elevation = 0.0 });
            if (!effectiveLevels.Any(l => l.Name == "Level 1"))
                effectiveLevels.Add(new LevelCommand
                {
                    Name      = "Level 1",
                    Elevation = effectiveLevels.Max(l => l.Elevation) + DEFAULT_STOREY_HEIGHT_MM,
                });

            foreach (var levelCmd in effectiveLevels)
            {
                // Skip if the template already contains a level with this name
                bool exists = new FilteredElementCollector(doc)
                    .OfClass(typeof(Level))
                    .Cast<Level>()
                    .Any(x => x.Name.Equals(levelCmd.Name, StringComparison.OrdinalIgnoreCase));

                if (exists) continue;

                Level level = Level.Create(doc, levelCmd.Elevation * MM_TO_FEET);
                level.Name = levelCmd.Name;
            }
        }

        // ------------------------------------------------------------------
        // 2. Structural grid lines  (Revit 2023: Grid.Create)
        // ------------------------------------------------------------------

        private void CreateGrids(List<GridCommand>? grids)
        {
            if (grids == null || grids.Count == 0) return;

            foreach (var gridCmd in grids)
            {
                if (gridCmd.Start == null || gridCmd.End == null) continue;

                XYZ startPt = new XYZ(
                    gridCmd.Start.X * MM_TO_FEET,
                    gridCmd.Start.Y * MM_TO_FEET,
                    0.0   // Grids live at Z = 0 (model origin)
                );
                XYZ endPt = new XYZ(
                    gridCmd.End.X * MM_TO_FEET,
                    gridCmd.End.Y * MM_TO_FEET,
                    0.0
                );

                // Revit requires the grid line to have a non-zero length
                if (startPt.DistanceTo(endPt) < 1e-6) continue;

                Line gridLine = Line.CreateBound(startPt, endPt);
                Grid grid = Grid.Create(doc, gridLine);

                if (!string.IsNullOrWhiteSpace(gridCmd.Name))
                {
                    try { grid.Name = gridCmd.Name; }
                    catch { /* duplicate name — leave auto-assigned name */ }
                }
            }
        }

        // ------------------------------------------------------------------
        // 3. Walls
        // ------------------------------------------------------------------

        private void CreateWalls(List<WallCommand>? walls)
        {
            if (walls == null) return;
            foreach (var wallCmd in walls)
            {
                WallType? wallType = GetWallType(wallCmd.Parameters?.WallType);
                if (wallType == null) continue;

                string levelName = wallCmd.Parameters?.Level ?? "Level 0";
                Level? level = GetLevel(levelName);
                if (level == null) continue;

                XYZ startPoint = new XYZ(
                    wallCmd.Parameters!.Curve.Start.X * MM_TO_FEET,
                    wallCmd.Parameters.Curve.Start.Y  * MM_TO_FEET,
                    wallCmd.Parameters.Curve.Start.Z  * MM_TO_FEET
                );
                XYZ endPoint = new XYZ(
                    wallCmd.Parameters.Curve.End.X * MM_TO_FEET,
                    wallCmd.Parameters.Curve.End.Y * MM_TO_FEET,
                    wallCmd.Parameters.Curve.End.Z * MM_TO_FEET
                );

                if (startPoint.DistanceTo(endPoint) < 1e-6) continue;

                Line wallLine = Line.CreateBound(startPoint, endPoint);

                Wall wall = Wall.Create(
                    doc,
                    wallLine,
                    wallType.Id,
                    level.Id,
                    wallCmd.Parameters.Height * MM_TO_FEET,
                    wallCmd.Parameters.Offset * MM_TO_FEET,
                    wallCmd.Parameters.Flip,
                    wallCmd.Parameters.Structural
                );

                SetWallProperties(wall, wallCmd.Properties);
            }
        }

        // ------------------------------------------------------------------
        // 4. Doors
        // ------------------------------------------------------------------

        private void CreateDoors(List<DoorCommand>? doors)
        {
            if (doors == null) return;
            foreach (var doorCmd in doors)
            {
                FamilySymbol? doorSymbol = GetFamilySymbol(
                    doorCmd.Parameters?.Family,
                    doorCmd.Parameters?.Symbol);
                if (doorSymbol == null) continue;
                if (!doorSymbol.IsActive) doorSymbol.Activate();

                Wall? hostWall = GetElementById<Wall>(doorCmd.Parameters?.HostWallId);
                if (hostWall == null) continue;

                string levelName = doorCmd.Parameters?.Level ?? "Level 0";
                Level? level = GetLevel(levelName);
                if (level == null) continue;

                XYZ location = new XYZ(
                    doorCmd.Parameters!.Location.X * MM_TO_FEET,
                    doorCmd.Parameters.Location.Y  * MM_TO_FEET,
                    doorCmd.Parameters.Location.Z  * MM_TO_FEET
                );

                FamilyInstance door = doc.Create.NewFamilyInstance(
                    location,
                    doorSymbol,
                    hostWall,
                    level,
                    StructuralType.NonStructural
                );

                if (doorCmd.Parameters.Rotation != 0)
                {
                    Line axis = Line.CreateBound(location, location + XYZ.BasisZ);
                    ElementTransformUtils.RotateElement(
                        doc, door.Id, axis,
                        doorCmd.Parameters.Rotation * Math.PI / 180.0);
                }
            }
        }

        // ------------------------------------------------------------------
        // 5. Windows
        // ------------------------------------------------------------------

        private void CreateWindows(List<WindowCommand>? windows)
        {
            if (windows == null) return;
            foreach (var winCmd in windows)
            {
                FamilySymbol? winSymbol = GetFamilySymbol(
                    winCmd.Parameters?.Family,
                    winCmd.Parameters?.Symbol);
                if (winSymbol == null) continue;
                if (!winSymbol.IsActive) winSymbol.Activate();

                Wall? hostWall = GetElementById<Wall>(winCmd.Parameters?.HostWallId);
                if (hostWall == null) continue;

                string levelName = winCmd.Parameters?.Level ?? "Level 0";
                Level? level = GetLevel(levelName);
                if (level == null) continue;

                XYZ location = new XYZ(
                    winCmd.Parameters!.Location.X * MM_TO_FEET,
                    winCmd.Parameters.Location.Y  * MM_TO_FEET,
                    winCmd.Parameters.Location.Z  * MM_TO_FEET
                );

                doc.Create.NewFamilyInstance(
                    location,
                    winSymbol,
                    hostWall,
                    level,
                    StructuralType.NonStructural
                );
            }
        }

        // ------------------------------------------------------------------
        // 6. Columns
        // ------------------------------------------------------------------

        private void CreateColumns(List<ColumnCommand>? columns)
        {
            if (columns == null) return;
            foreach (var colCmd in columns)
            {
                FamilySymbol? colSymbol = GetFamilySymbol(
                    colCmd.Parameters?.Family,
                    colCmd.Parameters?.Symbol);
                if (colSymbol == null) continue;
                if (!colSymbol.IsActive) colSymbol.Activate();

                string baseLevelName = colCmd.Parameters?.Level ?? "Level 0";
                Level? baseLevel = GetLevel(baseLevelName);
                if (baseLevel == null) continue;

                XYZ location = new XYZ(
                    colCmd.Parameters!.Location.X * MM_TO_FEET,
                    colCmd.Parameters.Location.Y  * MM_TO_FEET,
                    colCmd.Parameters.Location.Z  * MM_TO_FEET
                );

                FamilyInstance column = doc.Create.NewFamilyInstance(
                    location,
                    colSymbol,
                    baseLevel,
                    StructuralType.Column
                );

                // Set top level from transaction data — never hard-coded
                string topLevelName = colCmd.Parameters.TopLevel ?? "Level 1";
                Level? topLevel = GetLevel(topLevelName);
                if (topLevel != null)
                {
                    column.get_Parameter(BuiltInParameter.FAMILY_TOP_LEVEL_PARAM)
                          ?.Set(topLevel.Id);
                }

                if (colCmd.Parameters.Rotation != 0)
                {
                    Line axis = Line.CreateBound(location, location + XYZ.BasisZ);
                    ElementTransformUtils.RotateElement(
                        doc, column.Id, axis,
                        colCmd.Parameters.Rotation * Math.PI / 180.0);
                }
            }
        }

        // ------------------------------------------------------------------
        // 7. Floors
        // ------------------------------------------------------------------

        private void CreateFloors(List<FloorCommand>? floors)
        {
            if (floors == null) return;
            foreach (var floorCmd in floors)
            {
                if (floorCmd.Parameters?.Boundary == null
                    || floorCmd.Parameters.Boundary.Count < 3) continue;

                CurveArray profile = new CurveArray();
                int n = floorCmd.Parameters.Boundary.Count;
                for (int i = 0; i < n; i++)
                {
                    var p1 = floorCmd.Parameters.Boundary[i];
                    var p2 = floorCmd.Parameters.Boundary[(i + 1) % n];
                    profile.Append(Line.CreateBound(
                        new XYZ(p1.X * MM_TO_FEET, p1.Y * MM_TO_FEET, 0.0),
                        new XYZ(p2.X * MM_TO_FEET, p2.Y * MM_TO_FEET, 0.0)
                    ));
                }

                FloorType? floorType = GetFloorType(floorCmd.Parameters.FloorType);
                if (floorType == null) continue;

                string levelName = floorCmd.Parameters.Level ?? "Level 0";
                Level? level = GetLevel(levelName);
                if (level == null) continue;

#pragma warning disable CS0618  // NewFloor is deprecated in Revit 2024+ but present in 2023
                doc.Create.NewFloor(profile, floorType, level, floorCmd.Parameters.Structural);
#pragma warning restore CS0618
            }
        }

        // ------------------------------------------------------------------
        // 8. Rooms
        // ------------------------------------------------------------------

        private void CreateRooms(List<RoomCommand>? rooms)
        {
            if (rooms == null) return;
            foreach (var roomCmd in rooms)
            {
                string levelName = roomCmd.Parameters?.Level ?? "Level 0";
                Level? level = GetLevel(levelName);
                if (level == null) continue;

                UV point = new UV(
                    roomCmd.Parameters!.Point.X * MM_TO_FEET,
                    roomCmd.Parameters.Point.Y  * MM_TO_FEET
                );

                Room room = doc.Create.NewRoom(level, point);
                room.Name   = roomCmd.Parameters.Name;
                room.Number = roomCmd.Parameters.Number;
            }
        }

        // ------------------------------------------------------------------
        // 9. Views
        // ------------------------------------------------------------------

        private void CreateViews(List<ViewCommand>? views)
        {
            if (views == null) return;
            foreach (var viewCmd in views)
            {
                if (viewCmd.Parameters?.ViewType != "3D") continue;

                ViewFamilyType? vft = new FilteredElementCollector(doc)
                    .OfClass(typeof(ViewFamilyType))
                    .Cast<ViewFamilyType>()
                    .FirstOrDefault(x => x.ViewFamily == ViewFamily.ThreeDimensional);

                if (vft == null) continue;

                View3D view3d = View3D.CreatePerspective(doc, vft.Id);
                if (!string.IsNullOrWhiteSpace(viewCmd.Parameters.Name))
                    view3d.Name = viewCmd.Parameters.Name;

                // Realistic display style (6 = Realistic in Revit 2023)
                view3d.get_Parameter(BuiltInParameter.MODEL_GRAPHICS_STYLE)?.Set(6);
            }
        }

        // ------------------------------------------------------------------
        // Element lookup helpers
        // ------------------------------------------------------------------

        private T? GetElementById<T>(string? id) where T : Element
        {
            if (string.IsNullOrEmpty(id)) return null;
            if (!long.TryParse(id, out long eid)) return null;
            return doc.GetElement(new ElementId(eid)) as T;
        }

        private WallType? GetWallType(string? name)
        {
            if (string.IsNullOrEmpty(name)) return null;
            return new FilteredElementCollector(doc)
                .OfClass(typeof(WallType))
                .Cast<WallType>()
                .FirstOrDefault(x => x.Name.Equals(name, StringComparison.OrdinalIgnoreCase));
        }

        private FloorType? GetFloorType(string? name)
        {
            if (string.IsNullOrEmpty(name)) return null;
            return new FilteredElementCollector(doc)
                .OfClass(typeof(FloorType))
                .Cast<FloorType>()
                .FirstOrDefault(x => x.Name.Equals(name, StringComparison.OrdinalIgnoreCase));
        }

        private Level? GetLevel(string? name)
        {
            if (string.IsNullOrEmpty(name)) return null;
            return new FilteredElementCollector(doc)
                .OfClass(typeof(Level))
                .Cast<Level>()
                .FirstOrDefault(x => x.Name.Equals(name, StringComparison.OrdinalIgnoreCase));
        }

        private FamilySymbol? GetFamilySymbol(string? familyName, string? symbolName)
        {
            if (string.IsNullOrEmpty(familyName) || string.IsNullOrEmpty(symbolName))
                return null;
            return new FilteredElementCollector(doc)
                .OfClass(typeof(FamilySymbol))
                .Cast<FamilySymbol>()
                .FirstOrDefault(x =>
                    x.Family?.Name.Equals(familyName, StringComparison.OrdinalIgnoreCase) == true
                    && x.Name.Equals(symbolName, StringComparison.OrdinalIgnoreCase));
        }

        private void SetWallProperties(Wall wall, WallProperties? props)
        {
            if (props == null) return;
            // Additional property setting can be added here as needed
            // e.g. wall.get_Parameter(BuiltInParameter.WALL_FUNCTION)?.Set(...)
        }
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Transaction data model
    // All length values coming from the Python side are in MILLIMETRES.
    // ModelBuilder converts them to internal Revit feet using MM_TO_FEET.
    // ══════════════════════════════════════════════════════════════════════════

    public class RevitTransaction
    {
        public List<LevelCommand>  Levels  { get; set; } = new();
        public List<GridCommand>   Grids   { get; set; } = new();
        public List<WallCommand>   Walls   { get; set; } = new();
        public List<DoorCommand>   Doors   { get; set; } = new();
        public List<WindowCommand> Windows { get; set; } = new();
        public List<ColumnCommand> Columns { get; set; } = new();
        public List<FloorCommand>  Floors  { get; set; } = new();
        public List<RoomCommand>   Rooms   { get; set; } = new();
        public List<ViewCommand>   Views   { get; set; } = new();
    }

    // ── Levels ────────────────────────────────────────────────────────────────
    public class LevelCommand
    {
        /// <summary>Display name, e.g. "Level 0", "Level 1".</summary>
        public string Name      { get; set; } = string.Empty;
        /// <summary>Elevation above project base point in millimetres.</summary>
        public double Elevation { get; set; }
    }

    // ── Structural grid lines ─────────────────────────────────────────────────
    public class GridCommand
    {
        /// <summary>Grid bubble label, e.g. "1", "A".</summary>
        public string    Name  { get; set; } = string.Empty;
        public PointData Start { get; set; } = new();
        public PointData End   { get; set; } = new();
    }

    // ── Walls ─────────────────────────────────────────────────────────────────
    public class WallCommand
    {
        public WallParameters Parameters { get; set; } = new();
        public WallProperties  Properties { get; set; } = new();
    }

    public class WallParameters
    {
        public CurveData Curve      { get; set; } = new();
        public string    WallType   { get; set; } = string.Empty;
        /// <summary>Name of the base level, e.g. "Level 0".</summary>
        public string    Level      { get; set; } = "Level 0";
        /// <summary>Wall height in millimetres.</summary>
        public double    Height     { get; set; } = 2800;
        /// <summary>Base offset in millimetres.</summary>
        public double    Offset     { get; set; } = 0;
        public bool      Flip       { get; set; } = false;
        public bool      Structural { get; set; } = false;
    }

    public class WallProperties
    {
        public string Function   { get; set; } = "Interior";
        public string FireRating { get; set; } = "none";
    }

    // ── Shared geometry ───────────────────────────────────────────────────────
    public class CurveData
    {
        public PointData Start { get; set; } = new();
        public PointData End   { get; set; } = new();
    }

    public class PointData
    {
        /// <summary>Coordinate in millimetres.</summary>
        public double X { get; set; }
        public double Y { get; set; }
        public double Z { get; set; }
    }

    // ── Doors ─────────────────────────────────────────────────────────────────
    public class DoorCommand { public DoorParameters Parameters { get; set; } = new(); }

    public class DoorParameters
    {
        public string    Family     { get; set; } = string.Empty;
        public string    Symbol     { get; set; } = string.Empty;
        public PointData Location   { get; set; } = new();
        public string    HostWallId { get; set; } = string.Empty;
        /// <summary>Name of the base level, e.g. "Level 0".</summary>
        public string    Level      { get; set; } = "Level 0";
        /// <summary>Rotation in degrees.</summary>
        public double    Rotation   { get; set; } = 0;
    }

    // ── Windows ───────────────────────────────────────────────────────────────
    public class WindowCommand { public WindowParameters Parameters { get; set; } = new(); }

    public class WindowParameters
    {
        public string    Family     { get; set; } = string.Empty;
        public string    Symbol     { get; set; } = string.Empty;
        public PointData Location   { get; set; } = new();
        public string    HostWallId { get; set; } = string.Empty;
        public string    Level      { get; set; } = "Level 0";
    }

    // ── Columns ───────────────────────────────────────────────────────────────
    public class ColumnCommand
    {
        public ColumnParameters Parameters { get; set; } = new();
        public ColumnProperties  Properties { get; set; } = new();
    }

    public class ColumnParameters
    {
        public string    Family   { get; set; } = string.Empty;
        public string    Symbol   { get; set; } = string.Empty;
        public PointData Location { get; set; } = new();
        /// <summary>Base level name, e.g. "Level 0".</summary>
        public string    Level    { get; set; } = "Level 0";
        /// <summary>Top level name, e.g. "Level 1".  Never hard-coded.</summary>
        public string    TopLevel { get; set; } = "Level 1";
        /// <summary>Column height in millimetres (used only when top level is absent).</summary>
        public double    Height   { get; set; } = 2800;
        /// <summary>Rotation in degrees.</summary>
        public double    Rotation { get; set; } = 0;
    }

    public class ColumnProperties
    {
        public double Width    { get; set; } = 300;
        public double Depth    { get; set; } = 300;
        public string Material { get; set; } = "Concrete";
    }

    // ── Floors ────────────────────────────────────────────────────────────────
    public class FloorCommand { public FloorParameters Parameters { get; set; } = new(); }

    public class FloorParameters
    {
        public List<PointData> Boundary  { get; set; } = new();
        public string          FloorType { get; set; } = string.Empty;
        public string          Level     { get; set; } = "Level 0";
        public bool            Structural { get; set; } = false;
    }

    // ── Rooms ─────────────────────────────────────────────────────────────────
    public class RoomCommand { public RoomParameters Parameters { get; set; } = new(); }

    public class RoomParameters
    {
        public string    Name   { get; set; } = string.Empty;
        public string    Number { get; set; } = string.Empty;
        public string    Level  { get; set; } = "Level 0";
        public PointData Point  { get; set; } = new();
    }

    // ── Views ─────────────────────────────────────────────────────────────────
    public class ViewCommand { public ViewParameters Parameters { get; set; } = new(); }

    public class ViewParameters
    {
        public string ViewType { get; set; } = "3D";
        public string Name     { get; set; } = "3D View";
    }
}
