# Real lunar image pairs

No real Chandrayaan-2 pair is included in this repository. Obtain optical payload products through the [ISRO/ISSDC PRADAN Chandrayaan-2 archive](https://pradan.issdc.gov.in/ch2/). Sign-in, product availability, permitted use, registration, and acknowledgement requirements can change; confirm the current terms directly on PRADAN before downloading or publishing data. PRADAN data remain subject to ISRO copyright and the archive's terms.

A suitable independent reference is an LROC WAC lunar mosaic available through NASA's [LRO data products](https://science.nasa.gov/mission/lro/data-products/) or the Planetary Data System. Confirm the selected product's projection, resolution, and use terms before pairing it with an ISRO image; reproject/crop both images to overlapping terrain where appropriate.

Place each pair under these folders with the same basename and supported image extension:

```text
data/real/
  reference/
    site_001.png
  source/
    site_001.png
```

The evaluator pairs files by basename (extension may differ). Optional sun metadata can be supplied in `data/real/sun_angles.json`, mapping a pair basename to `{"reference": [azimuth_deg, elevation_deg], "source": [azimuth_deg, elevation_deg]}`. The current evaluator records these values alongside direct-pipeline results; it does not infer missing angles or create bridge images.

Run `python scripts/evaluate_real_pairs.py --data-dir data/real`. Results are written to `data/real/results.csv`. Do not commit restricted source imagery or metadata unless its terms allow redistribution.
