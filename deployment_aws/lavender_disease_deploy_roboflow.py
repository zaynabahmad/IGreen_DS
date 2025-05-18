import os
from PIL import Image
from roboflow import Roboflow
from IPython.display import display

# Initialize Roboflow API
rf = Roboflow(api_key="MHOwGJ6gNlwpudPBxOmc")
project = rf.workspace().project("lavender-disease")
model = project.version(1).model

# Path to the image file
image_file = "new_images/images (1).jpeg"

# Perform prediction on the image
prediction = model.predict(image_file, confidence=40, overlap=30)

# Print prediction results
print(prediction.json())

# Ensure the predictions folder exists
if not os.path.exists("predictions"):
    os.makedirs("predictions")

# Save the prediction image with the original image's name + 'predicted'
image_name = os.path.basename(image_file)  # Get the original image's name
prediction_image_path = f"predictions/{os.path.splitext(image_name)[0]}_predicted.jpg"  # Add '_predicted' to the name

# Save the prediction image
prediction.save(prediction_image_path)

# Display the saved prediction image
predicted_image = Image.open(prediction_image_path)
display(predicted_image)

print(f"Prediction image saved to {prediction_image_path}")
