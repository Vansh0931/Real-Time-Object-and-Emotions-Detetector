# IVP Pro — Viva Questions and Answers

1. **What is the aim?** Build a desktop system for live object detection/tracking and image-processing demonstrations.
2. **What is IVP?** Image and Video Processing: acquiring, transforming, analyzing and interpreting images and video frames.
3. **What is YOLO?** A family of object-detection models that predicts classes and bounding boxes efficiently.
4. **Why YOLO Nano?** It is relatively lightweight for demonstrations; speed depends on hardware.
5. **What is object detection?** Identifying object classes and their spatial locations.
6. **What is tracking?** Associating detections across frames to maintain temporary track IDs.
7. **What is ByteTrack?** A method for associating object detections across video frames.
8. **What is a bounding box?** A rectangle represented by coordinates around a detected object.
9. **What is a confidence threshold?** A cutoff used to filter low-scoring detections.
10. **What is FPS?** Frames processed per second, a throughput measure.
11. **What is latency?** Processing time measured in milliseconds.
12. **What is a Haar cascade?** A cascade classifier using Haar-like features; OpenCV supplies a pretrained face cascade.
13. **What is expression estimation?** Predicting a visual expression category from a face crop; it does not prove internal emotional state.
14. **Why use emojis?** They provide a compact visual label for the predicted category.
15. **What is FER+?** A facial-expression dataset/model family with categories such as happiness, sadness, anger and neutral.
16. **What is grayscale conversion?** Converting color channels into a single intensity channel.
17. **What is Gaussian blur?** A smoothing filter that reduces high-frequency variation and noise.
18. **What is Canny edge detection?** An edge detector based on image gradients and thresholding.
19. **What is a histogram?** A frequency distribution of pixel intensity values.
20. **Why use CSV logs?** They support later inspection, spreadsheet analysis and reproducible session records.
21. **What is a session report?** A summary of session duration, model, performance and detection-event counts.
22. **How does the alert work?** It compares current-frame detection count with a threshold and applies a cooldown.
23. **What is precision?** Correct positive predictions divided by all positive predictions.
24. **What is recall?** Correct positive predictions divided by all ground-truth positives.
25. **What is mAP50?** Mean average precision at an IoU threshold of 0.50.
26. **What is mAP50–95?** Mean average precision averaged across IoU thresholds 0.50 to 0.95.
27. **Can webcam-only output produce mAP?** No; labeled ground truth and a validation procedure are required.
28. **What affects performance?** Lighting, occlusion, input resolution, model size and computer hardware.
29. **Why use a background thread?** To keep the GUI responsive while frames are captured and inference runs.
30. **What are limitations?** Only trained classes are detected; tracking IDs can switch; expression estimates may be inaccurate; metrics need labeled data.
31. **What are future enhancements?** Custom training, video-file input, GPU acceleration, annotated recording and richer analytics.
