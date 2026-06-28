const express = require('express');
const path = require('path');
const fs = require('fs');
const readline = require('readline');

const app = express();
const PORT = process.env.PORT || 3000;

const PROCESSED_DIR = path.join(__dirname, '..', 'processed');

// Serve public directory
app.use(express.static(path.join(__dirname, 'public')));

// List datasets
app.get('/api/datasets', (req, res) => {
    if (!fs.existsSync(PROCESSED_DIR)) {
        return res.json([]);
    }
    try {
        const files = fs.readdirSync(PROCESSED_DIR)
            .filter(f => f.endsWith('.jsonl'))
            .map(f => {
                const stats = fs.statSync(path.join(PROCESSED_DIR, f));
                return {
                    name: f,
                    displayName: f.replace('minititan_', '').replace('.jsonl', '').replace('_', ' ').toUpperCase(),
                    sizeBytes: stats.size,
                    modifiedAt: stats.mtime
                };
            });
        res.json(files);
    } catch (err) {
        res.status(500).json({ error: 'Failed to read datasets directory' });
    }
});

// Get paginated dataset samples
app.get('/api/dataset/:name', async (req, res) => {
    const filename = req.params.name;
    const filePath = path.join(PROCESSED_DIR, filename);

    if (!filename.endsWith('.jsonl') || !fs.existsSync(filePath)) {
        return res.status(404).json({ error: 'Dataset not found' });
    }

    const page = parseInt(req.query.page) || 1;
    const limit = parseInt(req.query.limit) || 10;
    const query = (req.query.query || '').trim().toLowerCase();

    const startIdx = (page - 1) * limit;
    const endIdx = startIdx + limit;

    const fileStream = fs.createReadStream(filePath);
    const rl = readline.createInterface({
        input: fileStream,
        crlfDelay: Infinity
    });

    const results = [];
    let currentIdx = 0;
    let totalMatched = 0;

    try {
        for await (const line of rl) {
            if (!line.trim()) continue;
            try {
                const sample = JSON.parse(line);
                
                // Search query filter
                let matches = true;
                if (query) {
                    const rawString = line.toLowerCase();
                    matches = rawString.includes(query);
                }

                if (matches) {
                    if (totalMatched >= startIdx && totalMatched < endIdx) {
                        results.push({
                            index: currentIdx, // original index in file
                            matchIndex: totalMatched, // index in filtered list
                            data: sample
                        });
                    }
                    totalMatched++;
                }
                currentIdx++;
            } catch (e) {
                // Ignore corrupted lines
            }
        }

        res.json({
            dataset: filename,
            page,
            limit,
            totalItems: totalMatched,
            totalPages: Math.ceil(totalMatched / limit),
            items: results
        });
    } catch (err) {
        res.status(500).json({ error: 'Failed to stream dataset' });
    }
});

app.listen(PORT, () => {
    console.log(`Dataset Explorer running at http://localhost:${PORT}`);
});
