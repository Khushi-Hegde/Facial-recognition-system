# Face Recognition Identification System

A Python project that identifies enrolled people from images or a live
webcam.

## Features

-   Enroll people with face samples
-   Detect faces using YuNet
-   Extract face embeddings using SFace
-   Match faces using cosine similarity
-   Show `Unknown` when a face does not meet the matching threshold
-   Recognize faces from uploaded images and live webcam
-   Delete enrolled people

## Models Used

-   **Face Detection:** YuNet (`face_detection_yunet_2023mar.onnx`)
-   **Face Recognition:** SFace (`face_recognition_sface_2021dec.onnx`)

## Matching Threshold

The configured cosine similarity threshold is **0.45**.

## How It Works

1.  Enroll a person and save their face embedding.
2.  Detect a face in an uploaded image or webcam frame.
3.  Extract its embedding using SFace.
4.  Compare it with enrolled embeddings using cosine similarity.
5.  Show the person's name if the score meets the threshold; otherwise
    show `Unknown`.

## Installation

1.  Clone or download this repository.

2.  Create and activate a Python virtual environment.

3.  Install dependencies:

    ``` bash
    pip install -r requirements.txt
    ```

4.  Make sure the two ONNX model files are in the `Models/` folder.

## Run

``` bash
streamlit run app.py
```

Open the local URL shown in the terminal, usually
`http://localhost:8501`.

## Evaluation

Tested with **2 known-person images** and **1 unknown-person image**.

  Test                            Correct   Total   Observed Rate
  ----------------------------- --------- ------- ---------------
  Known-person identification           2       2            100%
  Unknown-person rejection              1       1            100%
  Overall on these test cases           3       3            100%

These results are based on only three test cases and do not establish
real-world accuracy.

## Failure Cases

No failure was observed in the three tests. The test set was small, so
more testing is needed. Possible cases to test include poor lighting,
face angles, low-quality images, and false matches.

## Improvements

-   Test with more known and unknown faces.
-   Test different lighting and face angles.
-   Compare thresholds using a separate validation set.
-   Record false matches and false rejections.
-   Protect stored face images and embeddings.

## Cost

The project uses free software and local models; no paid API or cloud
service is required for the described workflow. Existing computer,
internet, and electricity costs are not included.

## Privacy

Use only with consent. Do not upload enrolled face images or embeddings
to a public repository. This is an educational project, not a secure
authentication system.

## Author

**Khushi S Hegde**\
B.E. Artificial Intelligence and Data Science\
SDM Institute of Technology, Ujire

## References

-   [OpenCV Zoo](https://github.com/opencv/opencv_zoo)
-   [OpenCV Documentation](https://docs.opencv.org/)
-   [Streamlit Documentation](https://docs.streamlit.io/)
