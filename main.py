import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
import tensorflow as tf
from model import load_trained_model, get_label_mapping

# Define constants
IMG_SIZE = 32  # Ensure this matches the input size used during training
model = load_trained_model()
label_to_class = get_label_mapping()

DRAWN_IMAGE_PATH = os.getenv("DRAWN_IMAGE_PATH", "canvas.png")



def draw_expression(save_path=DRAWN_IMAGE_PATH, 
                    canvas_height=400, 
                    canvas_width=800,
                    line_thickness=15,
                    auto_process=True):
    """
    Opens an OpenCV window where the user can draw an expression.
    Controls:
    - Press 's' to save the current drawing (overwrites the save_path file if it exists)
    - Press 'c' to clear the canvas
    - Press 'q' to quit without saving further

    Parameters:
    - save_path: Path where the drawn image will be saved on pressing 's'.
    - canvas_height, canvas_width: Size of the drawing canvas.
    - line_thickness: Thickness of the brush stroke.
    - auto_process: If True, immediately segment, predict, and evaluate after saving.
    
    Returns:
    - save_path if the user saves an image,
    - None if the user quits without saving.
    """
    drawing = False
    ix, iy = -1, -1
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8)*255

    def draw_line(event, x, y, flags, param):
        nonlocal ix, iy, drawing, canvas
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            cv2.line(canvas, (ix, iy), (x, y), (0,0,0), line_thickness)
            ix, iy = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            cv2.line(canvas, (ix, iy), (x, y), (0,0,0), line_thickness)

    cv2.namedWindow('Draw', cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback('Draw', draw_line)
    print("Draw your expression in the window.")
    print("Instructions:")
    print("  s: Save")
    print("  c: Clear")
    print("  q: Quit without saving")

    # Add instructions overlay on canvas
    instructions_text = "Draw: LMB drag | s:Save | c:Clear | q:Quit"
    cv2.putText(canvas, instructions_text, (10, canvas_height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 1, cv2.LINE_AA)

    saved = False
    while True:
        cv2.imshow('Draw', canvas)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            # Quit without saving
            break
        elif k == ord('c'):
            # Clear the canvas
            canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8)*255
            cv2.putText(canvas, instructions_text, (10, canvas_height - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 1, cv2.LINE_AA)
        elif k == ord('s'):
            # Save the current drawing
            cv2.imwrite(save_path, canvas)
            print(f"Saved as {save_path}")
            saved = True
            if auto_process:
                # Immediately segment, predict, and evaluate
                symbol_imgs = segment_symbols(save_path)
                predictions = predict_symbols(symbol_imgs)
                for i, (label, probs) in enumerate(predictions):
                    print(f"Symbol {i}: {label}, Probs: {probs}")
                evaluate_expression(predictions)
            break

    cv2.destroyAllWindows()
    return save_path if saved else None

def segment_symbols(image_path, min_area=100):
    """
    Segment symbols from the input image.
    Steps:
    - Convert to grayscale
    - Otsu's threshold + dilation to find clear contours
    - Identify bounding boxes for symbols
    - Extract symbols from the *original grayscale image* to preserve training data similarity
    - Return a list of grayscale symbol images sorted by x-coordinate
    """
    original_img = cv2.imread(image_path)
    gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

    # Otsu’s threshold to get a binary image for contour detection
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Dilation to make symbols more contiguous
    kernel = np.ones((3,3), np.uint8)
    th = cv2.dilate(th, kernel, iterations=1)

    # Find contours on the thresholded image
    contours, hierarchy = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    symbol_images = []
    bounding_boxes = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w*h > min_area:
            # Extract the symbol region from the *original grayscale* image
            symbol = gray[y:y+h, x:x+w]
            symbol_images.append((x, symbol))
            bounding_boxes.append((x, y, w, h))

    # Sort by x-coordinate (left to right)
    symbol_images = sorted(symbol_images, key=lambda tup: tup[0])
    symbol_images = [im for _, im in symbol_images]

    # # Optional: Display bounding boxes for debugging
    # # (Remove if running headless)
    # for (x, y, w, h) in bounding_boxes:
    #     cv2.rectangle(original_img, (x, y), (x+w, y+h), (0,255,0), 2)
    # cv2.imshow("Segmented Symbols", original_img)
    # print("Press any key to continue...")
    # cv2.waitKey(0)
    cv2.destroyAllWindows()

    return symbol_images


def preprocess_symbol(symbol_img):
    """
    Preprocess a single symbol image for prediction:
    - Resize to IMG_SIZE x IMG_SIZE
    - Normalize to [0,1]
    - Add channel dimension
    Assumes symbol_img is already grayscale and similar to training data format.
    """
    symbol_resized = cv2.resize(symbol_img, (IMG_SIZE, IMG_SIZE))
    symbol_resized = symbol_resized.astype('float32') / 255.0
    symbol_resized = np.expand_dims(symbol_resized, axis=-1)
    return symbol_resized

def predict_symbols(symbol_images):
    """
    Predict classes for each symbol image using the global model and label_to_class.
    Expects symbol_images to be grayscale as returned by segment_symbols.
    """
    predictions = []
    if not symbol_images:
        print("No symbols found to predict.")
        return predictions

    for i, sym in enumerate(symbol_images):
        # Preprocess symbol before prediction
        img_preprocessed = preprocess_symbol(sym)
        img_preprocessed = np.expand_dims(img_preprocessed, axis=0)  # shape (1, IMG_SIZE, IMG_SIZE, 1)

        pred = model.predict(img_preprocessed, verbose=0)
        class_idx = np.argmax(pred)
        class_label = label_to_class[class_idx]
        predictions.append((class_label, pred[0]))
    return predictions



########################################
# Evaluate Expression
########################################

def evaluate_expression(predictions):
    """
    Takes predicted symbols and attempts to evaluate the mathematical expression.
    The expression is formed by mapping predicted labels (e.g., 'add', 'sub') to operators.
    Then tries to evaluate the resulting expression.
    """
    operator_map = {
        'add': '+',
        'sub': '-',
        'mul': '*',
        'div': '/'
    }

    def label_to_symbol(lbl):
        # If lbl is an operator, map it, otherwise assume it's a digit
        return operator_map.get(lbl, lbl)

    expression_str = "".join([label_to_symbol(lbl) for lbl, _ in predictions])
    print("Recognized Expression:", expression_str)

    expr = expression_str

    # # Safety: Only evaluate if consists of digits and allowed operators
    # # For simplicity, we trust input here, but a safer approach would parse manually
    # allowed_chars = set('0123456789+-*/')
    # if not all(ch in allowed_chars for ch in expr):
    #     print("Unsafe expression detected. Evaluation aborted.")
    #     return None

    try:
        result = eval(expr)
        print(f"Evaluation of {expr} = {result}")
        return result
    except Exception as e:
        print("Error evaluating expression:", e)
        return None
    


def main():
    # Let user draw an expression
    draw_expression(save_path="canvas.png", canvas_height=800, canvas_width=1000, line_thickness=20, auto_process=False)
    save_path = "canvas.png"
    # Segment symbols
    symbol_imgs = segment_symbols(save_path)

    # Save each segmented symbol for verification
    print("Saving segmented symbols to disk for verification...")
    for i, sym_img in enumerate(symbol_imgs):
        save_path = f"seg_symbol_{i}.png"
        cv2.imwrite(save_path, sym_img)
        print(f"Saved {save_path}")

    # Predict symbols one by one
    # Note: predict_symbols already loops through each symbol image individually
    predictions = predict_symbols(symbol_imgs)
    for i, (label, probs) in enumerate(predictions):
        print(f"Symbol {i}: {label}, Probs: {probs}")

    # Evaluate expression
    evaluate_expression(predictions)

    
if __name__ == "__main__":
    main()

