## Run the server:

```bash
uvicorn app.main:app --reload
```

## Run within a docker:

```bash
docker build . -t fastapi-service
docker run -p 8080:8080 fastapi-service
```

## Response bodies:

```
/images/single-inference
```

```json
[
    {
        "predicted_class": int,
        "probs":, float[],
    }
]
```


```
/images/batch-inference
```

```json
[
    {
        "image_name": str,
        "predicted_class": int,
        "probs":, float[],
    }
]
```
