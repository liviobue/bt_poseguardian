import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pymongo
import time

# Connect to MongoDB
client = pymongo.MongoClient("mongodb+srv://livio:Zuerich578@cluster0.axdzry5.mongodb.net/test")
db = client["bt_poseguardian"]
collection = db["data"]

# Fetch multiple frames
docs = list(collection.find().limit(10))

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Set plot limits (assume normalization between 0-1)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(-1, 1)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# Animation loop
for doc in docs:
    ax.clear()
    keypoints = doc["keypoints"]
    x_vals = [kp["x"] for kp in keypoints]
    y_vals = [kp["y"] for kp in keypoints]
    z_vals = [kp["z"] for kp in keypoints]

    ax.scatter(x_vals, y_vals, z_vals, c="r", marker="o")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(-1, 1)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("MediaPipe 3D Keypoints Over Time")

    plt.pause(0.5)  # Pause between frames

plt.show()
