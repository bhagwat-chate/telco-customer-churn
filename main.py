import os
from fastapi import FastAPI, Response
from starlette.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from uvicorn import run as app_run

from source.utility.utility import generate_global_timestamp
from source.entity.config_entity import PipelineConfig
from source.logger import setup_logger, logging
from source.pipeline.pipeline import DataPipeline
from source.constant.constant import APP_PORT, APP_HOST

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Function to run the pipeline
def run_pipeline(pipeline_type):
    try:
        global_timestamp = generate_global_timestamp()
        setup_logger(global_timestamp)  # Initialize logger with timestamp

        pipeline_obj = DataPipeline(global_timestamp)

        logging.info(f"START: MODEL {pipeline_type.upper()}")

        if pipeline_type == 'training':
            pipeline_obj.run_train_pipeline()
        elif pipeline_type == 'prediction':
            pipeline_obj.run_predict_pipeline()

        logging.info(f"END: MODEL {pipeline_type.upper()}")
        return f"Model {pipeline_type} complete"
    except Exception as e:
        logging.error(f"Error occurred: {e}", exc_info=True)
        return f"Error occurred: {e}"

@app.on_event("startup")
async def startup_event():
    logging.info("API startup")

@app.get("/", tags=['authentication'])
async def index():
    return RedirectResponse(url="/docs")

@app.get("/train", tags=['pipeline'])
async def train_route():
    result = run_pipeline("training")
    return {'message': result}


@app.get("/predict", tags=['pipeline'])
async def predict_route():
    result = run_pipeline("prediction")
    return {"message": result}

def main(pipeline_type):
    try:
        global_timestamp = generate_global_timestamp()
        setup_logger(global_timestamp)
        run_pipeline(pipeline_type)
    except Exception as e:
        print(e)
        logging.error(e, exc_info=True)


if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)
