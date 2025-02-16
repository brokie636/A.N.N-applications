# Task 3: Application of ANN (Artificial Neural Networks)

This repository contains the backend code and additional files for the **Application of ANN** task. The project involves developing and deploying neural network models for demand prediction, product classification, and personalized recommendations using FastAPI.

## Team Members
- Carolina Álvarez Murillo: [caroAM22](https://github.com/caroAM22)
- Alejandro Orozco Ochoa: [brokie636](https://github.com/brokie636)
- Juan José Zapata Cadavid: [jzapataca](https://github.com/jzapataca)

## Execution Instructions

1. Clone the repository: `git clone https://github.com/brokie636/A.N.N-applications.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the FastAPI server: `uvicorn main:app --reload` This will start the FastAPI application on http://127.0.0.1:8000.
   
## Additional Information

### Endpoints Overview
This backend exposes the following endpoints:

- Demand Prediction: Predicts sales for a range of days (1 to 30) or for a specific day based on multiple features such as temperature, fuel price, discounts, CPI, unemployment rate, store size, and holiday status.
- Product Classification: Classifies products based on uploaded images, helping organize inventory efficiently.

You can check out the detailed blog post related to this project [here](https://www.notion.so/Aplicaciones-de-Redes-Neuronales-Artificiales-196da9d4b08880cf9c2debad99ae75cf).
The backend is deployed and can be accessed [here](https://predicciones-9fuy.onrender.com).
