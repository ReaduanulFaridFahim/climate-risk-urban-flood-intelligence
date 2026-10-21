-- =========================================================================
-- Climate Risk & Urban Flood Intelligence for Bangladesh (Chattogram Study)
-- Star-Schema Data Warehouse DDL (PostgreSQL / DuckDB Compatible)
-- =========================================================================

-- Dimension 1: DimDate
CREATE TABLE IF NOT EXISTS DimDate (
    DateKey INT PRIMARY KEY,
    Date DATE NOT NULL,
    Year INT NOT NULL,
    Quarter VARCHAR(4) NOT NULL,
    Month INT NOT NULL,
    MonthName VARCHAR(15) NOT NULL,
    Day INT NOT NULL,
    DayOfWeek VARCHAR(12) NOT NULL,
    DayOfWeekNumber INT NOT NULL,
    IsWeekend BOOLEAN NOT NULL,
    Season VARCHAR(30) NOT NULL,
    HydrologicalYear INT NOT NULL
);

-- Dimension 2: DimArea
CREATE TABLE IF NOT EXISTS DimArea (
    AreaKey INT PRIMARY KEY,
    WardID VARCHAR(10) NOT NULL,
    WardName VARCHAR(50) NOT NULL,
    Thana VARCHAR(50) NOT NULL,
    AreaSqKm DECIMAL(6, 2) NOT NULL,
    MeanElevationM DECIMAL(5, 2) NOT NULL,
    MaxElevationM DECIMAL(5, 2) NOT NULL,
    MeanSlopeDeg DECIMAL(4, 2) NOT NULL,
    BaselineImperviousRatio DECIMAL(4, 3) NOT NULL,
    DistanceToRiverKm DECIMAL(4, 2) NOT NULL,
    GeomorphicTerrainClass VARCHAR(60) NOT NULL,
    Latitude DECIMAL(8, 5) NOT NULL,
    Longitude DECIMAL(8, 5) NOT NULL
);

-- Dimension 3: DimEvent
CREATE TABLE IF NOT EXISTS DimEvent (
    EventKey INT PRIMARY KEY,
    EventName VARCHAR(80) NOT NULL,
    StartDate DATE NOT NULL,
    EndDate DATE NOT NULL,
    EventType VARCHAR(40) NOT NULL,
    DurationDays INT NOT NULL,
    SeverityClass VARCHAR(20) NOT NULL,
    PeakRainfallMm DECIMAL(6, 2) NOT NULL,
    MeteorologicalNotes TEXT
);

-- Fact 1: FactRainfall (NASA GPM IMERG)
CREATE TABLE IF NOT EXISTS FactRainfall (
    RainfallKey INT PRIMARY KEY,
    Year INT NOT NULL,
    Month INT NOT NULL,
    DateKey INT REFERENCES DimDate(DateKey),
    MonthlyRainfallMm DECIMAL(7, 2) NOT NULL,
    MaxDailyRainfallRx1dayMm DECIMAL(6, 2) NOT NULL,
    HeavyRainfallDaysR50mm INT NOT NULL,
    ConsecutiveDryDaysCDD INT NOT NULL,
    PrecipitationAnomalyPercentage DECIMAL(6, 2) NOT NULL,
    SatelliteSource VARCHAR(50) NOT NULL,
    BMDStationValidationBiasMm DECIMAL(5, 2) NOT NULL
);

-- Fact 2: FactLandCover (Multi-Temporal Satellite LULC)
CREATE TABLE IF NOT EXISTS FactLandCover (
    LandCoverKey INT PRIMARY KEY,
    Year INT NOT NULL,
    AreaKey INT REFERENCES DimArea(AreaKey),
    BuiltUpSqKm DECIMAL(6, 3) NOT NULL,
    VegetationSqKm DECIMAL(6, 3) NOT NULL,
    WaterBodiesSqKm DECIMAL(6, 3) NOT NULL,
    BareSoilHillsSqKm DECIMAL(6, 3) NOT NULL,
    ImperviousSurfacePercentage DECIMAL(5, 2) NOT NULL,
    VegetationCoveragePercentage DECIMAL(5, 2) NOT NULL,
    NaturalDrainageLossIndex DECIMAL(4, 3) NOT NULL,
    ClassificationAccuracyKappa DECIMAL(4, 3) NOT NULL
);

-- Fact 3: FactFloodExtent (Observed Satellite SAR Inundation)
CREATE TABLE IF NOT EXISTS FactFloodExtent (
    FloodExtentKey INT PRIMARY KEY,
    EventKey INT REFERENCES DimEvent(EventKey),
    AreaKey INT REFERENCES DimArea(AreaKey),
    InundatedAreaSqKm DECIMAL(6, 3) NOT NULL,
    InundatedPercentageOfWard DECIMAL(5, 2) NOT NULL,
    MeanInundationDepthM DECIMAL(4, 2) NOT NULL,
    StagnantDurationHours INT NOT NULL,
    WaterRecessionRateCmPerHr DECIMAL(5, 2) NOT NULL,
    PrimaryMechanism VARCHAR(50) NOT NULL,
    SatelliteSensor VARCHAR(60) NOT NULL
);

-- Fact 4: FactExposure (WorldPop Population & Assets at Risk)
CREATE TABLE IF NOT EXISTS FactExposure (
    ExposureKey INT PRIMARY KEY,
    EventKey INT REFERENCES DimEvent(EventKey),
    AreaKey INT REFERENCES DimArea(AreaKey),
    TotalWardPopulation INT NOT NULL,
    ExposedPopulation INT NOT NULL,
    ExposedVulnerablePopulation INT NOT NULL,
    PopulationExposureRatePct DECIMAL(5, 2) NOT NULL,
    ExposedBuiltUpSqKm DECIMAL(6, 3) NOT NULL,
    EstimatedDirectDamageUSD_Millions DECIMAL(7, 2) NOT NULL,
    CompoundRiskIndex DECIMAL(6, 2) NOT NULL,
    DataConfidenceTier VARCHAR(80) NOT NULL
);

-- Analytical Indices for Zero-Latency Power BI DirectQuery
CREATE INDEX idx_rain_date ON FactRainfall(DateKey);
CREATE INDEX idx_lc_area ON FactLandCover(AreaKey);
CREATE INDEX idx_flood_event_area ON FactFloodExtent(EventKey, AreaKey);
CREATE INDEX idx_exp_event_area ON FactExposure(EventKey, AreaKey);
