# Soccer_analysis
Multi camera tracking and analysis of soccer match

- [x] Prototyping
- [ ] Optimization
- [ ] Production


# Todo list

- [x] Players detection
  - [x] divide images into multiple smallest sub-images
  - [x] use TF Object Detection API for players detection on each sub-image
  - [x] transpose player position from sub-image to image
- [x] Projection on playground
  - [x] calibration for homography
  - [x] perform homography on each camera frame to transpose players position
  - [x] merge projections of the same player from different cameras
- [ ] Players tracking
  - [ ] do more research


- [ ] Software
  - [ ] software design
  - [ ] Create class Detected_object

# Optimization

- [ ] Players detection
  - [ ] Try other network architecture
  - [x] remove overlapped bboxes
  - [ ] overlap between sub-images
  - [ ] constrain ratio for bbox /!\ domain specific
- [ ] Projection on playground
  - [ ] find other merge/clustering algorithms
- [ ] Players tracking
  - [ ] use optical flow
- [ ] Software
  - [ ] multi GPUs (horovod)
