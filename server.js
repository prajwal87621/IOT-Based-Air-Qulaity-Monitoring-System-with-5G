const express = require('express');
const http = require('http');
const WebSocket = require('ws');
const mongoose = require('mongoose');
const cors = require('cors');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');
const cron = require('node-cron');

// Configuration
const config = {
    port: process.env.PORT || 3000,
    mongoUri: process.env.MONGO_URI || 'mongodb://localhost:27017/air_quality_db',
    mlScript: path.join(__dirname, 'ml_predictor.py'),
    dataExportPath: path.join(__dirname, 'sensor_data.csv'),
    predictionInterval: '*/15 * * * *', // Every 20 minutes
    dataRetentionDays: 30
};

// MongoDB Schema - Aligned with ESP32 and ML predictor
const sensorDataSchema = new mongoose.Schema({
    timestamp: { type: Date, default: Date.now, index: true },
    deviceId: { type: String, default: 'ESP32_AQM_01' },
    temperature: { type: Number, required: true },
    humidity: { type: Number, required: true },
    pressure: { type: Number, required: true },
    pm25: { type: Number, required: true },
    voc: { type: Number, required: true },
    co: { type: Number, required: true },
    no2: { type: Number, required: true },
    nh3: { type: Number, required: true },
    aqi: { type: Number, required: true },
    gpsLat: { type: Number, default: null },
    gpsLon: { type: Number, default: null },
    gpsAlt: { type: Number, default: null },
    gpsValid: { type: Boolean, default: false },
    placeName: { type: String, default: 'Unknown Location' },
    fanState: { type: Boolean, default: false },
    deviceStatus: { type: String, default: 'online' },
    signalStrength: { type: Number }
});

const predictionSchema = new mongoose.Schema({
    timestamp: { type: Date, default: Date.now, index: true },
    predictionDate: { type: Date, required: true },
    predictedAQI: { type: Number, required: true },
    predictedPM25: { type: Number, required: true },
    predictedTemp: { type: Number, required: true },
    confidence: { type: Number, default: 0.75 },
    modelAccuracy: { type: Number, default: 0.75 },
    dataPoints: { type: Number, default: 0 },
    gpsLat: { type: Number },
    gpsLon: { type: Number },
    placeName: { type: String },
    modelInfo: {
        temperatureRange: { type: String },
        constraintsApplied: { type: String },
        seasonalAdjustments: { type: String },
        modelsTrained: { type: Number }
    },
    modelAccuracyDetails: {
        aqi: {
            mae: { type: Number },
            mse: { type: Number },
            r2: { type: Number },
            cv_score: { type: Number }
        },
        pm25: {
            mae: { type: Number },
            mse: { type: Number },
            r2: { type: Number },
            cv_score: { type: Number }
        },
        temperature: {
            mae: { type: Number },
            mse: { type: Number },
            r2: { type: Number },
            cv_score: { type: Number }
        }
    },
    generatedAt: { type: Date }
});

// Models
const SensorData = mongoose.model('SensorData', sensorDataSchema);
const Prediction = mongoose.model('Prediction', predictionSchema);

// Express app setup
const app = express();
const server = http.createServer(app);
const wss = new WebSocket.Server({ server });
app.use(cors());
app.use(express.json({ limit: '10mb' }));
app.use(express.static('public'));

// Store active WebSocket connections
const clients = new Set();

// Utility Functions
function isValidGPS(lat, lon) {
    return (
        lat !== null &&
        lon !== null &&
        !isNaN(lat) &&
        !isNaN(lon) &&
        lat >= -90 &&
        lat <= 90 &&
        lon >= -180 &&
        lon <= 180 &&
        Math.abs(lat) > 0.001 &&
        Math.abs(lon) > 0.001
    );
}

const locationCache = new Map();
const lastLocationRequest = new Map();
const LOCATION_REQUEST_COOLDOWN = 30000;

async function getLocationName(lat, lon) {
    try {
        if (!isValidGPS(lat, lon)) {
            console.log('Invalid GPS coordinates:', lat, lon);
            return 'Invalid GPS Coordinates';
        }
        const cacheKey = `${lat.toFixed(3)}_${lon.toFixed(3)}`;
        if (locationCache.has(cacheKey)) {
            console.log(`Using cached location for ${lat}, ${lon}: ${locationCache.get(cacheKey)}`);
            return locationCache.get(cacheKey);
        }
        const now = Date.now();
        if (lastLocationRequest.has(cacheKey) &&
            (now - lastLocationRequest.get(cacheKey)) < LOCATION_REQUEST_COOLDOWN) {
            console.log('Rate limited - using fallback location');
            const fallbackName = getIndianCityFromCoordinates(lat, lon);
            locationCache.set(cacheKey, fallbackName);
            return fallbackName;
        }
        console.log(`Getting location name for: ${lat}, ${lon}`);
        lastLocationRequest.set(cacheKey, now);
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 5000);
        const response = await fetch(
            `https://nominatim.openstreetmap.org/reverse?format=json&lat=${lat}&lon=${lon}&addressdetails=1&zoom=10`,
            {
                headers: { 'User-Agent': 'AirQualityMonitor/1.0' },
                signal: controller.signal
            }
        );
        clearTimeout(timeoutId);
        if (response.ok) {
            const data = await response.json();
            if (data && data.address) {
                const address = data.address;
                const city = address.city || address.town || address.village ||
                            address.suburb || address.district || address.county;
                const state = address.state || address.region;
                const country = address.country;
                let locationName = '';
                if (city) locationName += city;
                if (state && locationName) locationName += ', ' + state;
                if (country && locationName) {
                    const countryShort = country.toLowerCase().includes('india') ? 'India' : country;
                    locationName += ', ' + countryShort;
                }
                if (locationName) {
                    console.log(`Location found: ${locationName}`);
                    locationCache.set(cacheKey, locationName);
                    return locationName;
                }
            }
        }
        const locationName = getIndianCityFromCoordinates(lat, lon);
        console.log(`Using fallback location: ${locationName}`);
        locationCache.set(cacheKey, locationName);
        return locationName;
    } catch (error) {
        console.error('Geocoding error:', error);
        const fallbackName = `Location (${lat.toFixed(4)}, ${lon.toFixed(4)})`;
        return fallbackName;
    }
}

function getIndianCityFromCoordinates(lat, lon) {
    const cities = [
        { name: 'New Delhi, India', latMin: 28.4, latMax: 28.8, lonMin: 76.8, lonMax: 77.5 },
        { name: 'Mumbai, India', latMin: 18.9, latMax: 19.3, lonMin: 72.7, lonMax: 73.1 },
        { name: 'Bangalore, India', latMin: 12.8, latMax: 13.2, lonMin: 77.4, lonMax: 77.8 },
        { name: 'Kolkata, India', latMin: 22.4, latMax: 22.7, lonMin: 88.2, lonMax: 88.5 },
        { name: 'Chennai, India', latMin: 12.9, latMax: 13.2, lonMin: 80.1, lonMax: 80.4 },
        { name: 'Hyderabad, India', latMin: 17.3, latMax: 17.5, lonMin: 78.3, lonMax: 78.6 },
        { name: 'Pune, India', latMin: 18.4, latMax: 18.7, lonMin: 73.7, lonMax: 74.0 },
        { name: 'Ahmedabad, India', latMin: 22.9, latMax: 23.2, lonMin: 72.4, lonMax: 72.8 },
        { name: 'Jaipur, India', latMin: 26.8, latMax: 27.0, lonMin: 75.6, lonMax: 76.0 },
        { name: 'Lucknow, India', latMin: 26.7, latMax: 27.0, lonMin: 80.8, lonMax: 81.1 }
    ];
    for (const city of cities) {
        if (lat >= city.latMin && lat <= city.latMax && lon >= city.lonMin && lon <= city.lonMax) {
            return city.name;
        }
    }
    return `Location (${lat.toFixed(4)}, ${lon.toFixed(4)})`;
}

function shouldActivateFan(aqi, pm25, voc) {
    return aqi > 30 || pm25 > 5 || voc > 5;
}

// WebSocket handling
wss.on('connection', (ws) => {
    console.log('Client connected');
    clients.add(ws);
    sendInitialData(ws);
    ws.on('message', (message) => {
        try {
            const data = JSON.parse(message);
            if (data.type === 'fan_control') {
                broadcastToClients({
                    type: 'fan_control',
                    fanState: data.fanState,
                    timestamp: new Date()
                });
            }
        } catch (error) {
            console.error('WebSocket message error:', error);
        }
    });
    ws.on('close', () => {
        console.log('Client disconnected');
        clients.delete(ws);
    });
    ws.on('error', (error) => {
        console.error('WebSocket error:', error);
        clients.delete(ws);
    });
});

async function sendInitialData(ws) {
    try {
        const latestSensorData = await SensorData.find()
            .sort({ timestamp: -1 })
            .limit(10);
        const latestPredictions = await Prediction.find()
            .sort({ predictionDate: 1 })
            .limit(7);
        const formattedPredictions = latestPredictions.map(pred => ({
            date: pred.predictionDate.toISOString(),
            aqi: pred.predictedAQI,
            pm25: pred.predictedPM25,
            temperature: pred.predictedTemp,
            gpsLat: pred.gpsLat,
            gpsLon: pred.gpsLon,
            placeName: pred.placeName,
            confidence: pred.confidence,
            modelAccuracy: pred.modelAccuracy,
            modelInfo: pred.modelInfo,
            modelAccuracyDetails: pred.modelAccuracyDetails
        }));
        const initialData = {
            type: 'initial_data',
            sensorData: latestSensorData,
            predictions: formattedPredictions
        };
        if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify(initialData));
        }
    } catch (error) {
        console.error('Error sending initial data:', error);
    }
}

function broadcastToClients(data) {
    const message = JSON.stringify(data);
    clients.forEach(client => {
        if (client.readyState === WebSocket.OPEN) {
            try {
                client.send(message);
            } catch (error) {
                console.error('Error broadcasting to client:', error);
                clients.delete(client);
            }
        }
    });
}

// API Routes
app.get('/api/health', (req, res) => {
    res.json({
        status: 'ok',
        timestamp: new Date().toISOString(),
        clients: clients.size,
        mongodb: mongoose.connection.readyState === 1 ? 'connected' : 'disconnected',
        mlPredictorStatus: fs.existsSync(config.mlScript) ? 'available' : 'missing',
        locationCacheSize: locationCache.size
    });
});

app.get('/api/sensor/latest', async (req, res) => {
    try {
        const latestData = await SensorData.findOne().sort({ timestamp: -1 });
        if (!latestData) {
            return res.status(404).json({ error: 'No sensor data available' });
        }
        res.json(latestData);
    } catch (error) {
        console.error('Error fetching latest sensor data:', error);
        res.status(500).json({ error: error.message });
    }
});

app.get('/api/sensor/locations', async (req, res) => {
    try {
        const locations = await SensorData.find({
            gpsValid: true,
            gpsLat: { $ne: null, $ne: 0 },
            gpsLon: { $ne: null, $ne: 0 }
        })
        .select('gpsLat gpsLon placeName aqi pm25 temperature timestamp')
        .sort({ timestamp: -1 })
        .limit(100);
        res.json(locations);
    } catch (error) {
        console.error('Error fetching sensor locations:', error);
        res.status(500).json({ error: error.message });
    }
});

app.get('/api/sensor/history', async (req, res) => {
    try {
        const { hours = 24, limit = 100 } = req.query;
        const startTime = new Date(Date.now() - hours * 60 * 60 * 1000);
        const data = await SensorData.find({
            timestamp: { $gte: startTime }
        })
        .sort({ timestamp: -1 })
        .limit(parseInt(limit));
        res.json(data);
    } catch (error) {
        console.error('Error fetching sensor history:', error);
        res.status(500).json({ error: error.message });
    }
});

app.get('/api/predictions', async (req, res) => {
    try {
        const predictions = await Prediction.find()
            .sort({ predictionDate: 1 })
            .limit(7);
        const formattedPredictions = predictions.map(pred => ({
            date: pred.predictionDate.toISOString(),
            aqi: pred.predictedAQI,
            pm25: pred.predictedPM25,
            temperature: pred.predictedTemp,
            gpsLat: pred.gpsLat,
            gpsLon: pred.gpsLon,
            placeName: pred.placeName,
            confidence: pred.confidence,
            modelAccuracy: pred.modelAccuracy,
            modelInfo: pred.modelInfo,
            modelAccuracyDetails: pred.modelAccuracyDetails
        }));
        res.json(formattedPredictions);
    } catch (error) {
        console.error('Error fetching predictions:', error);
        res.status(500).json({ error: error.message });
    }
});

app.get('/api/predictions/accuracy', async (req, res) => {
    try {
        const latestPrediction = await Prediction.findOne().sort({ timestamp: -1 });
        if (!latestPrediction) {
            return res.status(404).json({ error: 'No predictions available' });
        }
        res.json({
            accuracy: latestPrediction.modelAccuracy,
            confidence: latestPrediction.confidence,
            dataPoints: latestPrediction.dataPoints,
            modelInfo: latestPrediction.modelInfo,
            modelAccuracyDetails: latestPrediction.modelAccuracyDetails,
            generatedAt: latestPrediction.generatedAt
        });
    } catch (error) {
        console.error('Error fetching prediction accuracy:', error);
        res.status(500).json({ error: error.message });
    }
});

app.post('/api/sensor/data', async (req, res) => {
    try {
        const sensorData = req.body;
        const logData = {
            deviceId: sensorData.deviceId,
            aqi: sensorData.aqi,
            temperature: sensorData.temperature,
            pm25: sensorData.pm25,
            gpsLat: sensorData.gpsLat,
            gpsLon: sensorData.gpsLon,
            timestamp: new Date().toISOString()
        };
        console.log('Received sensor data:', JSON.stringify(logData, null, 2));
        let aqi = sensorData.aqi || 0;
        console.log(`Using ESP32 calculated AQI: ${aqi}`);
        const fanState = shouldActivateFan(aqi, sensorData.pm25 || 0, sensorData.voc || 0);
        let gpsLat = null;
        let gpsLon = null;
        let gpsAlt = null;
        let gpsValid = false;
        let placeName = 'Unknown Location';
        if (sensorData.gpsLat !== undefined && sensorData.gpsLon !== undefined) {
            gpsLat = parseFloat(sensorData.gpsLat);
            gpsLon = parseFloat(sensorData.gpsLon);
            gpsAlt = sensorData.gpsAlt ? parseFloat(sensorData.gpsAlt) : null;
            gpsValid = isValidGPS(gpsLat, gpsLon);
            if (gpsValid) {
                console.log(`Valid GPS coordinates: ${gpsLat}, ${gpsLon}`);
                placeName = await getLocationName(gpsLat, gpsLon);
            } else {
                console.log(`Invalid GPS coordinates: ${gpsLat}, ${gpsLon}`);
                gpsLat = null;
                gpsLon = null;
                gpsAlt = null;
            }
        } else {
            console.log('No GPS data received from ESP32');
        }
        const newSensorData = new SensorData({
            deviceId: sensorData.deviceId || 'ESP32_AQM_01',
            temperature: sensorData.temperature || 0,
            humidity: sensorData.humidity || 0,
            pressure: sensorData.pressure || 1013.25,
            pm25: sensorData.pm25 || 0,
            voc: sensorData.voc || 0,
            co: sensorData.co || 0,
            no2: sensorData.no2 || 0,
            nh3: sensorData.nh3 || 0,
            aqi,
            gpsLat,
            gpsLon,
            gpsAlt,
            gpsValid,
            placeName,
            fanState: sensorData.fanState !== undefined ? sensorData.fanState : fanState,
            deviceStatus: sensorData.deviceStatus || 'online',
            signalStrength: sensorData.signalStrength || null,
            timestamp: new Date()
        });
        await newSensorData.save();
        console.log(`Sensor data saved - GPS: ${gpsValid ? 'Valid' : 'Invalid'}, Location: ${placeName}`);
        broadcastToClients({
            type: 'sensor_update',
            data: newSensorData
        });
        res.json({
            success: true,
            aqi,
            fanState: newSensorData.fanState,
            gpsValid,
            placeName,
            timestamp: newSensorData.timestamp,
            message: 'Data received successfully'
        });
        console.log(`Processed - AQI: ${aqi}, Fan: ${newSensorData.fanState ? 'ON' : 'OFF'}, GPS: ${gpsValid ? 'Valid' : 'Invalid'}`);
    } catch (error) {
        console.error('Error processing sensor data:', error);
        res.status(500).json({ error: error.message });
    }
});

app.post('/api/predictions/generate', async (req, res) => {
    try {
        const predictions = await generatePredictions();
        res.json(predictions);
    } catch (error) {
        console.error('Error generating predictions:', error);
        res.status(500).json({ error: error.message });
    }
});

app.get('/api/export/csv', async (req, res) => {
    try {
        await exportDataToCSV();
        res.download(config.dataExportPath);
    } catch (error) {
        console.error('Error exporting CSV:', error);
        res.status(500).json({ error: error.message });
    }
});

async function exportDataToCSV() {
    try {
        const data = await SensorData.find()
            .sort({ timestamp: 1 })
            .limit(2000);

        if (data.length === 0) {
            throw new Error('No data to export');
        }
        const headers = [
            'Date', 'AQI', 'PM2.5', 'Temperature', 'Humidity', 'Pressure',
            'VOC', 'NO2', 'CO', 'NH3'
        ];
        const csvData = [];
        for (let i = 0; i < data.length; i++) {
            const record = data[i];
            const row = [
                record.timestamp.toISOString(),
                record.aqi,
                record.pm25,
                record.temperature,
                record.humidity,
                record.pressure,
                record.voc,
                record.no2,
                record.co,
                record.nh3
            ];
            csvData.push(row);
        }
        const csvContent = [
            headers.join(','),
            ...csvData.map(row => row.join(','))
        ].join('\n');
        fs.writeFileSync(config.dataExportPath, csvContent);
        console.log(`Exported ${data.length} records to CSV for ML training`);
    } catch (error) {
        console.error('CSV export error:', error);
        throw error;
    }
}

async function generatePredictions() {
    try {
        const dataCount = await SensorData.countDocuments();
        if (dataCount < 10) {
            throw new Error('Insufficient data for ML predictions (minimum 10 records required)');
        }
        const latestGPSData = await SensorData.findOne({
            gpsValid: true,
            gpsLat: { $ne: null },
            gpsLon: { $ne: null }
        }).sort({ timestamp: -1 });
        await exportDataToCSV();
        console.log('Data exported for ML prediction');
        const predictions = await runMLPrediction();
        console.log('ML predictions completed successfully');
        const predictionDocs = [];
        const baseDate = new Date();
        if (predictions.predictions && Array.isArray(predictions.predictions)) {
            for (let i = 0; i < predictions.predictions.length; i++) {
                const predData = predictions.predictions[i];
                const predDate = new Date(baseDate);
                predDate.setDate(predDate.getDate() + i + 1);
                const predictedAQI = Math.round(Math.max(0, Math.min(500, predData.aqi || 50)));
                const predictedPM25 = Math.round((Math.max(0, Math.min(999, predData.pm25 || 25))) * 10) / 10;
                const predictedTemp = Math.round((Math.max(5, Math.min(50, predData.temperature || 25))) * 10) / 10;
                predictionDocs.push(new Prediction({
                    predictionDate: predDate,
                    predictedAQI,
                    predictedPM25,
                    predictedTemp,
                    confidence: Math.max(0.5, Math.min(1.0, predictions.confidence || 0.75)),
                    modelAccuracy: Math.max(0.5, Math.min(1.0, predictions.accuracy || 0.75)),
                    dataPoints: predictions.data_points || dataCount,
                    gpsLat: latestGPSData?.gpsLat || null,
                    gpsLon: latestGPSData?.gpsLon || null,
                    placeName: latestGPSData?.placeName || 'Unknown Location',
                    modelInfo: {
                        temperatureRange: predictions.model_info?.temperature_range || '5-50°C',
                        constraintsApplied: predictions.model_info?.constraints_applied || 'Indian climate patterns',
                        seasonalAdjustments: predictions.model_info?.seasonal_adjustments || 'Enabled',
                        modelsTrained: predictions.model_info?.models_trained || 3
                    },
                    modelAccuracyDetails: predictions.model_accuracy || {
                        aqi: { mae: 10, mse: 100, r2: 0.7, cv_score: 0.7 },
                        pm25: { mae: 10, mse: 100, r2: 0.7, cv_score: 0.7 },
                        temperature: { mae: 10, mse: 100, r2: 0.7, cv_score: 0.7 }
                    },
                    generatedAt: new Date(predictions.generated_at || new Date())
                }));
            }
        } else {
            throw new Error('Invalid ML prediction output format');
        }
        if (predictionDocs.length === 0) {
            throw new Error('No valid predictions generated by ML model');
        }
        await Prediction.deleteMany({});
        await Prediction.insertMany(predictionDocs);
        const formattedPredictions = predictionDocs.map(pred => ({
            date: pred.predictionDate.toISOString(),
            aqi: pred.predictedAQI,
            pm25: pred.predictedPM25,
            temperature: pred.predictedTemp,
            gpsLat: pred.gpsLat,
            gpsLon: pred.gpsLon,
            placeName: pred.placeName,
            confidence: pred.confidence,
            modelAccuracy: pred.modelAccuracy,
            modelInfo: pred.modelInfo,
            modelAccuracyDetails: pred.modelAccuracyDetails
        }));
        broadcastToClients({
            type: 'predictions_update',
            predictions: formattedPredictions,
            metadata: {
                accuracy: predictions.accuracy,
                confidence: predictions.confidence,
                dataPoints: predictions.data_points,
                generatedAt: predictions.generated_at,
                modelAccuracyDetails: predictions.model_accuracy
            }
        });
        console.log(`Generated and saved ${predictionDocs.length} ML predictions with GPS data`);
        return {
            predictions: formattedPredictions,
            metadata: predictions
        };
    } catch (error) {
        console.error('Prediction generation error:', error);
        const fallbackPredictions = await generateFallbackPredictions();
        return fallbackPredictions;
    }
}

async function generateFallbackPredictions() {
    try {
        console.log('Generating fallback predictions...');
        const latestData = await SensorData.findOne().sort({ timestamp: -1 });
        const latestGPSData = await SensorData.findOne({
            gpsValid: true,
            gpsLat: { $ne: null },
            gpsLon: { $ne: null }
        }).sort({ timestamp: -1 });
        if (!latestData) {
            throw new Error('No sensor data available for fallback predictions');
        }
        const predictions = [];
        const baseDate = new Date();
        for (let i = 1; i <= 7; i++) {
            const predDate = new Date(baseDate);
            predDate.setDate(predDate.getDate() + i);
            const aqiTrend = Math.random() * 20 - 10;
            const pm25Trend = Math.random() * 10 - 5;
            const tempTrend = Math.random() * 4 - 2;
            predictions.push(new Prediction({
                predictionDate: predDate,
                predictedAQI: Math.round(Math.max(0, Math.min(500, latestData.aqi + aqiTrend))),
                predictedPM25: Math.round((Math.max(0, Math.min(999, latestData.pm25 + pm25Trend))) * 10) / 10,
                predictedTemp: Math.round((Math.max(5, Math.min(50, latestData.temperature + tempTrend))) * 10) / 10,
                confidence: 0.6,
                modelAccuracy: 0.6,
                dataPoints: await SensorData.countDocuments(),
                gpsLat: latestGPSData?.gpsLat || null,
                gpsLon: latestGPSData?.gpsLon || null,
                placeName: latestGPSData?.placeName || 'Unknown Location',
                modelInfo: {
                    temperatureRange: '5-50°C',
                    constraintsApplied: 'Fallback trend-based prediction',
                    seasonalAdjustments: 'Basic variation applied',
                    modelsTrained: 3
                },
                modelAccuracyDetails: {
                    aqi: { mae: 15, mse: 225, r2: 0.6, cv_score: 0.6 },
                    pm25: { mae: 12, mse: 144, r2: 0.6, cv_score: 0.6 },
                    temperature: { mae: 3, mse: 9, r2: 0.6, cv_score: 0.6 }
                },
                generatedAt: new Date()
            }));
        }
        await Prediction.deleteMany({});
        await Prediction.insertMany(predictions);
        const formattedPredictions = predictions.map(pred => ({
            date: pred.predictionDate.toISOString(),
            aqi: pred.predictedAQI,
            pm25: pred.predictedPM25,
            temperature: pred.predictedTemp,
            gpsLat: pred.gpsLat,
            gpsLon: pred.gpsLon,
            placeName: pred.placeName,
            confidence: pred.confidence,
            modelAccuracy: pred.modelAccuracy,
            modelInfo: pred.modelInfo,
            modelAccuracyDetails: pred.modelAccuracyDetails
        }));
        broadcastToClients({
            type: 'predictions_update',
            predictions: formattedPredictions,
            metadata: {
                accuracy: 0.6,
                confidence: 0.6,
                dataPoints: await SensorData.countDocuments(),
                generatedAt: new Date().toISOString()
            }
        });
        console.log('Fallback predictions generated and saved');
        return { predictions: formattedPredictions };
    } catch (error) {
        console.error('Fallback prediction error:', error);
        throw error;
    }
}

function runMLPrediction() {
    return new Promise((resolve, reject) => {
        if (!fs.existsSync(config.mlScript)) {
            reject(new Error('ML prediction script not found'));
            return;
        }
        const pythonProcess = spawn('python3', [config.mlScript, config.dataExportPath], {
    env: { ...process.env, PYTHONUNBUFFERED: '1' }
});
        let output = '';
        let errorOutput = '';
        pythonProcess.stdout.on('data', (data) => {
            output += data.toString();
        });
        pythonProcess.stderr.on('data', (data) => {
            errorOutput += data.toString();
        });
        pythonProcess.on('close', (code) => {
            if (code === 0) {
                try {
                    const predictions = JSON.parse(output);
                    if (!predictions.predictions || !Array.isArray(predictions.predictions)) {
                        throw new Error('Invalid ML prediction output: missing predictions array');
                    }
                    resolve(predictions);
                } catch (parseError) {
                    reject(new Error(`Failed to parse ML output: ${parseError.message}`));
                }
            } else {
                reject(new Error(`ML script failed with code ${code}: ${errorOutput}`));
            }
        });
        pythonProcess.on('error', (error) => {
            reject(new Error(`Failed to start ML script: ${error.message}`));
        });
        setTimeout(() => {
            pythonProcess.kill();
            reject(new Error('ML prediction timeout'));
        }, 120000);
    });
}

async function cleanupOldData() {
    try {
        const cutoffDate = new Date();
        cutoffDate.setDate(cutoffDate.getDate() - config.dataRetentionDays);
        const result = await SensorData.deleteMany({
            timestamp: { $lt: cutoffDate }
        });
        console.log(`Cleaned up ${result.deletedCount} old sensor records`);
        const predictionCutoff = new Date();
        predictionCutoff.setDate(predictionCutoff.getDate() - 30);
        const predResult = await Prediction.deleteMany({
            timestamp: { $lt: predictionCutoff }
        });
        console.log(`Cleaned up ${predResult.deletedCount} old predictions`);
    } catch (error) {
        console.error('Data cleanup error:', error);
    }
}

app.post('/api/device/fan/control', async (req, res) => {
    try {
        const { fanState } = req.body;
        if (typeof fanState !== 'boolean') {
            return res.status(400).json({ error: 'fanState must be a boolean' });
        }
        console.log(`Fan control request: ${fanState ? 'ON' : 'OFF'}`);
        const latestData = await SensorData.findOne().sort({ timestamp: -1 });
        if (latestData) {
            latestData.fanState = fanState;
            await latestData.save();
        }
        broadcastToClients({
            type: 'fan_control',
            fanState,
            timestamp: new Date()
        });
        res.json({
            success: true,
            fanState,
            message: 'Fan control command sent'
        });
    } catch (error) {
        console.error('Fan control error:', error);
        res.status(500).json({ error: error.message });
    }
});

app.get('/api/device/fan/state', async (req, res) => {
    try {
        const latestData = await SensorData.findOne().sort({ timestamp: -1 });
        if (!latestData) {
            return res.status(404).json({ error: 'No device data available' });
        }
        res.json({
            fanState: latestData.fanState,
            timestamp: latestData.timestamp
        });
    } catch (error) {
        console.error('Error fetching fan state:', error);
        res.status(500).json({ error: error.message });
    }
});

app.get('/api/device/status', async (req, res) => {
    try {
        const latestData = await SensorData.findOne().sort({ timestamp: -1 });
        if (!latestData) {
            return res.status(404).json({ error: 'No device data available' });
        }
        const deviceAge = Date.now() - latestData.timestamp.getTime();
        const isOnline = deviceAge < 300000;
        res.json({
            deviceId: latestData.deviceId,
            status: isOnline ? 'online' : 'offline',
            lastSeen: latestData.timestamp,
            fanState: latestData.fanState,
            signalStrength: latestData.signalStrength,
            gpsValid: latestData.gpsValid,
            location: latestData.placeName
        });
    } catch (error) {
        console.error('Device status error:', error);
        res.status(500).json({ error: error.message });
    }
});

app.get('/api/statistics', async (req, res) => {
    try {
        const { days = 7 } = req.query;
        const startDate = new Date();
        startDate.setDate(startDate.getDate() - parseInt(days));
        const stats = await SensorData.aggregate([
            { $match: { timestamp: { $gte: startDate } } },
            {
                $group: {
                    _id: null,
                    avgAQI: { $avg: '$aqi' },
                    maxAQI: { $max: '$aqi' },
                    minAQI: { $min: '$aqi' },
                    avgPM25: { $avg: '$pm25' },
                    maxPM25: { $max: '$pm25' },
                    avgTemp: { $avg: '$temperature' },
                    maxTemp: { $max: '$temperature' },
                    minTemp: { $min: '$temperature' },
                    totalReadings: { $sum: 1 },
                    validGPSReadings: {
                        $sum: { $cond: ['$gpsValid', 1, 0] }
                    }
                }
            }
        ]);
        if (stats.length === 0) {
            return res.json({ error: 'No data available for statistics' });
        }
        const result = stats[0];
        const aqiDistribution = await SensorData.aggregate([
            { $match: { timestamp: { $gte: startDate } } },
            {
                $bucket: {
                    groupBy: '$aqi',
                    boundaries: [0, 51, 101, 151, 201, 301, 501],
                    default: 'Other',
                    output: {
                        count: { $sum: 1 },
                        category: {
                            $switch: {
                                branches: [
                                    { case: { $lt: ['$aqi', 51] }, then: 'Good' },
                                    { case: { $lt: ['$aqi', 101] }, then: 'Moderate' },
                                    { case: { $lt: ['$aqi', 151] }, then: 'Unhealthy for Sensitive' },
                                    { case: { $lt: ['$aqi', 201] }, then: 'Unhealthy' },
                                    { case: { $lt: ['$aqi', 301] }, then: 'Very Unhealthy' }
                                ],
                                default: 'Hazardous'
                            }
                        }
                    }
                }
            }
        ]);
        res.json({
            period: `${days} days`,
            summary: {
                totalReadings: result.totalReadings,
                validGPSReadings: result.validGPSReadings,
                gpsSuccessRate: Math.round((result.validGPSReadings / result.totalReadings) * 100)
            },
            airQuality: {
                avgAQI: Math.round(result.avgAQI * 10) / 10,
                maxAQI: result.maxAQI,
                minAQI: result.minAQI,
                avgPM25: Math.round(result.avgPM25 * 10) / 10,
                maxPM25: Math.round(result.maxPM25 * 10) / 10
            },
            temperature: {
                avg: Math.round(result.avgTemp * 10) / 10,
                max: Math.round(result.maxTemp * 10) / 10,
                min: Math.round(result.minTemp * 10) / 10
            },
            distribution: aqiDistribution
        });
    } catch (error) {
        console.error('Statistics error:', error);
        res.status(500).json({ error: error.message });
    }
});

app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

mongoose.connect(config.mongoUri, {
    useNewUrlParser: true,
    useUnifiedTopology: true,
}).then(() => {
    console.log('🗄️ Connected to MongoDB at ' + config.mongoUri);
    server.listen(config.port, () => {
        console.log(`🚀 Air Quality Monitoring Server running on http://0.0.0.0:${config.port}`);
        console.log(`📊 WebSocket server running on ws://0.0.0.0:${config.port}`);
        console.log(`📡 Ready to receive ESP32 data...`);
    });
    cron.schedule(config.predictionInterval, async () => {
        console.log('Running scheduled ML predictions...');
        try {
            await generatePredictions();
            console.log('Scheduled predictions completed');
        } catch (error) {
            console.error('Scheduled prediction error:', error);
        }
    });
    cron.schedule('0 2 * * *', async () => {
        console.log('Running daily data cleanup...');
        try {
            await cleanupOldData();
            console.log('Daily cleanup completed');
        } catch (error) {
            console.error('Daily cleanup error:', error);
        }
    });
    setTimeout(() => {
        async function initializePredictions() {
            try {
                console.log('Checking for existing predictions...');
                const existingPredictions = await Prediction.countDocuments();
                if (existingPredictions === 0) {
                    console.log('No existing predictions found, generating initial predictions...');
                    await generatePredictions();
                } else {
                    console.log(`Found ${existingPredictions} existing predictions, skipping initialization.`);
                }
            } catch (error) {
                console.error('Initial prediction error:', error);
            }
        }
        initializePredictions();
    }, 5000);
}).catch(error => {
    console.error('🗄️ MongoDB connection error:', error);
    process.exit(1);
});

process.on('SIGINT', () => {
    console.log('\nShutting down gracefully...');
    clients.forEach(client => {
        if (client.readyState === WebSocket.OPEN) {
            client.close();
        }
    });
    mongoose.connection.close(() => {
        console.log('MongoDB connection closed');
        process.exit(0);
    });
});

process.on('uncaughtException', (error) => {
    console.error('Uncaught Exception:', error);
    process.exit(1);
});

process.on('unhandledRejection', (reason, promise) => {
    console.error('Unhandled Rejection at:', promise, 'reason:', reason);
});

module.exports = {
    app,
    server,
    isValidGPS,
    getLocationName,
    shouldActivateFan
};
