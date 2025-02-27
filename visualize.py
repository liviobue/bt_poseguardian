import matplotlib.pyplot as plt
import pymongo

# Connect to MongoDB and fetch data
client = pymongo.MongoClient("mongodb+srv://livio:Zuerich578@cluster0.axdzry5.mongodb.net/test")
db = client["bt_poseguardian"]
collection = db["data"]

# Fetch multiple documents (e.g., first 10 frames)
docs = list(collection.find().limit(10))

plt.figure(figsize=(8, 6))

# Plot all frames with different colors
for i, doc in enumerate(docs):
    keypoints = doc["keypoints"]
    x_vals = [kp["x"] for kp in keypoints]
    y_vals = [kp["y"] for kp in keypoints]
    
    plt.scatter(x_vals, y_vals, label=f'Frame {i+1}', alpha=0.6)

plt.xlabel("X (Horizontal)")
plt.ylabel("Y (Vertical)")
plt.legend()
plt.gca().invert_yaxis()  # Invert Y to match image coordinate system
plt.title("Multiple Frames of MediaPipe Keypoints")
plt.show()
