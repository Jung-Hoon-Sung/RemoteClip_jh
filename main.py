from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from fastapi.responses import FileResponse
import zipfile
import os
import shutil
from remoteclip import RemoteClip, visualize_top_results
import json
from enum import Enum
from PIL import Image
from fastapi.responses import FileResponse

app = FastAPI()

def float32_converter_single(obj):
    if isinstance(obj, np.float32):
        return float(obj)
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')

def transform_json(original_json, file_name):
    return [item[0] for item in original_json[file_name]]

class ModelName(str, Enum):
    ViTL14 = "ViT-L-14"
    ViTB32 = "ViT-B-32"
    RN50 = "RN50"
    
class ImageUploadResponse(BaseModel):
    json: str
    image: str

@app.post("/Image Caption Single", response_model=ImageUploadResponse, 
          tags=["remoteclip"], 
          summary="Predict content of a single image", 
          description="Upload a single image. The server will predict the contents of the image.")
async def predict_single_image(image: UploadFile = File(..., description="A single image to be predicted."), 
                               model_name: ModelName = Form(..., description="The name of the model to use for prediction. Options are: 'RN50', 'ViT-B-32', 'ViT-L-14'.")):
    try:
        if not os.path.exists("images"):
            os.makedirs("images")
        for filename in os.listdir("images"):
            file_path = os.path.join("images", filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

        # Check if the uploaded file is an image
        if not any([image.filename.endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]]):
            return {
                "json": "Error",
                "image": "File should be an image (.jpg, .jpeg, .png, .JPG, .JPEG, .PNG)."
            }

        # Save the uploaded image
        temp_image_path = os.path.join("images", image.filename)
        with open(temp_image_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        # Load the model and predict keywords
        clip = RemoteClip(model_name.value)
        result = clip.predict(temp_image_path)
        
        # Visualize results
        visualize_top_results([image.filename], [Image.open(temp_image_path)], result, "visualized_results_single.png")

        transformed_result = transform_json(result, image.filename)
        
        # Save transformed results to JSON
        output_json_path = "predictions_single.json"
        with open(output_json_path, 'w') as json_file:
            json.dump(transformed_result, json_file, ensure_ascii=False, indent=4)

        # Prepare response with saved JSON and image
        return {
            "json": output_json_path,
            "image": "visualized_results_single.png"
        }
    except Exception as e:
        return {
            "json": "Error",
            "image": f"An error occurred: {str(e)}"
        }


@app.get("/download/visualized_results_single.png")
async def download_visualized_results_single():
    return FileResponse("visualized_results_single.png", media_type="image/png", filename="visualized_results_single.png")

@app.get("/download/predictions_single.json")
async def download_predictions_single():
    return FileResponse("predictions_single.json", media_type="application/json", filename="predictions_single.json")

###########################################################################################################################################################################


@app.post("/Image Caption Multi", response_model=ImageUploadResponse, 
          tags=["remoteclip"], 
          summary="Predict image contents", 
          description="Upload a .zip file containing images. The server will predict the contents of the images.")
async def predict_image(file: UploadFile = File(..., description="images.zip file containing one or more images to be predicted."), 
                        model_name: ModelName = Form(..., description="The name of the model to use for prediction. Options are: 'RN50', 'ViT-B-32', 'ViT-L-14'.")):
    try:
        
        if not file.filename.endswith(".zip"):
            return {
                "json": "Error",
                "image": "File should be a zip archive containing images."
            }

        # Save uploaded zip file
        temp_path = "temp_upload.zip"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Unzip the file
        with zipfile.ZipFile(temp_path, 'r') as zip_ref:
            zip_ref.extractall("images")
            
        os.remove(temp_path)  # Remove the uploaded zip file

        # Load the model and predict keywords
        clip = RemoteClip(model_name.value)
        
        results = {}
        image_names = os.listdir("images")
        local_images = [Image.open(os.path.join("images", image_name)) for image_name in image_names]
        
        for image_path in image_names:
            result = clip.predict(os.path.join("images", image_path))
            results.update(result)
        
        # Visualize results
        visualize_top_results(image_names, local_images, results, "visualized_results.png")
    
        # Save results to JSON
        output_json_path = "predictions.json"
        with open(output_json_path, 'w') as json_file:
            json.dump(results, json_file, ensure_ascii=False, indent=4)

        # Prepare response with saved JSON and image
        return {
            "json": output_json_path,
            "image": "visualized_results.png"
        }
    except Exception as e:
        return {
            "json": "Error",
            "image": f"An error occurred: {str(e)}"
        }

@app.get("/download/visualized_Multi_results.png")
async def download_visualized_results():
    return FileResponse("visualized_results.png", media_type="image/png", filename="visualized_results.png")

@app.get("/download/predictions_Multi.json")
async def download_predictions():
    return FileResponse("predictions.json", media_type="application/json", filename="predictions.json")