#Deployed on 3/20/25
#main.py
from fastapi import FastAPI
from data.general import load_all_data
from data.health_issue import load_recommendations_from_h5, preload_all_product_details
from routes.routes import router 

app = FastAPI(title="Health Advisor API")

# Include the router
app.include_router(router)

# Add CORS middleware here (if needed)
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

recommendations = load_recommendations_from_h5('recommendations.h5')
product_cache = preload_all_product_details()
load_all_data()

# ----------------- Main Loop -----------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

