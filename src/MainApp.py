import cv2

from src.models.RCNN import Rcnn

def main_app():
    # Open a connection with the camera
    cap = cv2.VideoCapture(0)

    # Create rcnn object
    rcnn = Rcnn()
    if not cap.isOpened():
        print("Nie udalo sie nawiazac polaczenia z kamera")
        exit(1)

    while True:
        ret, frame = cap.read()

        # Was the frame read correctly?
        if not ret:
            print("Nie udalo sie odczytac klatki")
            break

        frame = rcnn.search_image(frame)
        cv2.imshow("Podglad z kamery", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release of resources after work is done
    cap.release()
    cv2.destroyAllWindows()

main_app()