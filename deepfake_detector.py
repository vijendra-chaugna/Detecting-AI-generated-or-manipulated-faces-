import os
import cv2
import torch
import argparse
from PIL import Image
from facenet_pytorch import MTCNN
import torch.nn as nn
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

def extract_faces(video_dir, label, output_dir, mtcnn, frame_skip=5):
    """Extracts faces from videos in video_dir using MTCNN, saving to output_dir/label."""
    save_path = os.path.join(output_dir, label)
    os.makedirs(save_path, exist_ok=True)
    for fname in os.listdir(video_dir):
        if not fname.lower().endswith(('.mp4', '.avi', '.mov')):
            continue
        video_path = os.path.join(video_dir, fname)
        cap = cv2.VideoCapture(video_path)
        frame_id = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_id % frame_skip == 0:
                # Convert frame (BGR -> RGB) and detect faces
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                boxes, probs = mtcnn.detect(img_rgb)
                if boxes is not None:
                    # Take the first detected face (highest probability)
                    for box, prob in zip(boxes, probs):
                        if prob >= 0.70:
                            x1, y1, x2, y2 = [int(b) for b in box]
                            # Clamp coordinates to image boundaries
                            h, w, _ = frame.shape
                            x1, y1 = max(x1, 0), max(y1, 0)
                            x2, y2 = min(x2, w), min(y2, h)
                            if x2 <= x1 or y2 <= y1:
                                continue
                            face_crop = frame[y1:y2, x1:x2]
                            if face_crop.size == 0:
                                continue
                            face_img = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
                            face_img = face_img.resize((224, 224))
                            base = os.path.splitext(fname)[0]
                            out_fname = f"{base}_{frame_id}.jpg"
                            face_img.save(os.path.join(save_path, out_fname))
                            break  # only save one face per frame
            frame_id += 1
        cap.release()

def prepare_dataset(data_root, compression):
    """Prepares face dataset by extracting from videos if needed."""
    # Define paths based on FaceForensics++ structure
    real_vid_dir = os.path.join(data_root, 'original_sequences', 'youtube', compression, 'videos')
    fake_vid_dir = os.path.join(data_root, 'manipulated_sequences', 'Deepfakes', compression, 'videos')
    faces_dir = 'data'
    real_faces = os.path.join(faces_dir, 'real')
    fake_faces = os.path.join(faces_dir, 'fake')
    # Check if faces are already extracted
    if os.path.exists(real_faces) and os.path.exists(fake_faces):
        print("Face dataset already exists, skipping extraction.")
        return faces_dir
    print("Extracting faces from videos...")
    # Initialize MTCNN for face detection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(keep_all=False, device=device)
    os.makedirs(real_faces, exist_ok=True)
    os.makedirs(fake_faces, exist_ok=True)
    # Extract faces from real and fake videos
    extract_faces(real_vid_dir, 'real', faces_dir, mtcnn)
    extract_faces(fake_vid_dir, 'fake', faces_dir, mtcnn)
    print("Face extraction complete.")
    return faces_dir

def train_model(args):
    # Prepare face images dataset (extract faces if needed)
    faces_dir = prepare_dataset(args.data, args.compression)
    # Define transforms for input images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # Load dataset
    dataset = datasets.ImageFolder(faces_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    # Map class indices: ensure label 0=Real, 1=Fake
    # if dataset.classes == ['fake', 'real']:
    #     idx_map = {0:1, 1:0}
    # else:
    #     idx_map = {0:0, 1:1}
    # Initialize ResNet18 model (pretrained on ImageNet:contentReference[oaicite:3]{index=3})
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # binary classifier: real vs fake
    model = model.to(device)
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    print(f"Starting training for {args.epochs} epochs...")
    best_loss = float('inf')
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for imgs, labels in dataloader:
            # Remap labels if needed (real=0, fake=1)
            # labels = labels.clone()
            # for i in range(labels.size(0)):
            #     labels[i] = idx_map[int(labels[i])
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        epoch_loss = running_loss / len(dataset)
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {epoch_loss:.4f}")
        # Save best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), args.save_model)
    print(f"Training complete. Best model saved to {args.save_model}")

def predict_image(args):
    # Load the trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model = model.to(device).eval()
    # Detect face in the input image
    mtcnn = MTCNN(keep_all=False, device=device)
    img = Image.open(args.image).convert('RGB')
    face = mtcnn(img)
    if face is None:
        print("No face detected in the image.")
        return
    # Prepare face for model input

    face = face.permute(1, 2, 0).cpu().numpy()
    face = (face * 255).astype('uint8')   # convert correct range
    face = Image.fromarray(face)
    face = face.resize((224, 224))

    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ])
    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


    inp = transform(face).unsqueeze(0).to(device)
    # Predict
    with torch.no_grad():
        out = model(inp)
        probs = torch.softmax(out, dim=1).cpu().numpy()[0]
    fake_prob = probs[0]  # model's second output is "Fake"
    real_prob = probs[1]
    pred = "Fake" if fake_prob > real_prob else "Real"
    confidence = max(fake_prob, real_prob) * 100
    print(f"Result: {pred} ({confidence:.1f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deepfake Detection with ResNet18 and MTCNN")
    parser.add_argument('--mode', choices=['train', 'predict'], required=True,
                        help="train to fine-tune model, predict to classify an image")
    parser.add_argument('--data', type=str, default='./FaceForensics',
                        help="Root directory of FaceForensics++ dataset")
    parser.add_argument('--compression', type=str, default='raw',
                        choices=['raw', 'c23', 'c40'], help="Compression folder name in FaceForensics++")
    parser.add_argument('--image', type=str, help="Path to an image for prediction")
    parser.add_argument('--model', type=str, default='resnet18_deepfake.pth',
                        help="Path to save/load the trained model weights")
    parser.add_argument('--epochs', type=int, default=5, help="Number of training epochs")
    parser.add_argument('--batch-size', type=int, default=16, help="Training batch size")
    parser.add_argument('--save-model', type=str, default='resnet18_deepfake.pth',
                        help="File path to save the trained model")
    args = parser.parse_args()

    if args.mode == 'train':
        train_model(args)
    elif args.mode == 'predict':
        if not args.image:
            print("Please provide --image for prediction mode.")
        else:
            predict_image(args)
