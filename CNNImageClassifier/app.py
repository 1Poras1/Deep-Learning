

import os
import re
import zipfile
import shutil
import tempfile
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
from cnn_model import build_cnn
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    load_img,
    img_to_array,
)


st.set_page_config(
    page_title="CNN Image Classifier",
    page_icon="📸",
    layout="wide",
)

st.title("🧠 CNN Image Classifier")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ---------------- SESSION STATE ----------------
defaults = {
    "model": None,
    "img_size": 32,
    "class_names": [],
    "dataset_type": None,
    "dataset_temp_dir": None,
    "dataset_zip_name": None,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


def image_files(folder):
    """Return supported image files recursively."""
    return [
        path for path in Path(folder).rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def find_class_directory(root):
    """
    Find a directory containing exactly two class subfolders.
    Supports extra wrapper directories around Cat and Dog.
    """
    candidates = []

    for current, dirs, _ in os.walk(root):
        dirs[:] = [
            d for d in dirs
            if d not in {"__MACOSX", ".DS_Store"}
        ]

        class_dirs = [
            os.path.join(current, d)
            for d in dirs
            if image_files(os.path.join(current, d))
        ]

        if len(class_dirs) == 2:
            candidates.append((current, class_dirs))

    if not candidates:
        return None, None

    # Prefer the deepest directory with exactly two image-containing folders.
    candidates.sort(key=lambda item: len(Path(item[0]).parts), reverse=True)
    return candidates[0]


def make_filename_dataset(files, destination):
    """
    Organize mixed images only when filenames clearly identify
    each image as cat or dog.
    """
    cat_dir = Path(destination) / "Cat"
    dog_dir = Path(destination) / "Dog"
    cat_dir.mkdir(parents=True, exist_ok=True)
    dog_dir.mkdir(parents=True, exist_ok=True)

    copied = {"Cat": 0, "Dog": 0}
    ambiguous = 0

    for source in files:
        # Match names such as cat.123.jpg, dog_123.jpg,
        # cat.jpg and dog-image.png.
        name = source.stem.lower()

        is_cat = bool(re.search(r"(^|[^a-z])cat([^a-z]|$)", name))
        is_dog = bool(re.search(r"(^|[^a-z])dog([^a-z]|$)", name))

        # Also support common dataset names like cat.123 and dog.456.
        if not is_cat and not is_dog:
            if re.match(r"^cat[._ -]?\d+$", name):
                is_cat = True
            elif re.match(r"^dog[._ -]?\d+$", name):
                is_dog = True

        if is_cat == is_dog:
            ambiguous += 1
            continue

        label = "Cat" if is_cat else "Dog"
        target = Path(destination) / label / source.name

        # Avoid collisions between files with identical names.
        if target.exists():
            target = target.with_name(
                f"{source.parent.name}_{source.name}"
            )

        shutil.copy2(source, target)
        copied[label] += 1

    return copied, ambiguous


# ---------------- SIDEBAR ----------------
st.sidebar.header("🔍 Choose Section")
section = st.sidebar.radio("Go to:", ["Train Model", "Predict"])


# ---------------- TRAINING ----------------
if section == "Train Model":

    dataset_choice = st.sidebar.radio(
        "Dataset:",
        ["CIFAR-10", "Cats vs Dogs (ZIP Upload)"],
    )

    img_size = st.sidebar.slider(
        "Image Size", min_value=32, max_value=256, value=64, step=32
    )
    batch_size = st.sidebar.selectbox("Batch size", [16, 32, 64])
    epochs = st.sidebar.slider("Epochs", 1, 20, 5)

    # =============== CIFAR-10 ===============
    if dataset_choice == "CIFAR-10":
        st.subheader("📦 CIFAR-10 Dataset")

        if st.button("🧠 Train CNN on CIFAR-10"):
            with st.spinner("Loading CIFAR-10 and training..."):
                (x_train, y_train), (x_test, y_test) = cifar10.load_data()

                x_train = x_train.astype("float32") / 255.0
                x_test = x_test.astype("float32") / 255.0

                y_train_cat = to_categorical(y_train, 10)
                y_test_cat = to_categorical(y_test, 10)

                model = build_cnn((32, 32, 3), 10)

                history = model.fit(
                    x_train,
                    y_train_cat,
                    validation_data=(x_test, y_test_cat),
                    epochs=epochs,
                    batch_size=batch_size,
                )

            st.session_state.model = model
            st.session_state.img_size = 32
            st.session_state.class_names = [
                "airplane", "automobile", "bird", "cat", "deer",
                "dog", "frog", "horse", "ship", "truck",
            ]
            st.session_state.dataset_type = "CIFAR-10"

            st.success("CIFAR-10 training complete!")

            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
            ax[0].plot(history.history["accuracy"], label="Train")
            ax[0].plot(history.history["val_accuracy"], label="Validation")
            ax[0].set_title("Accuracy")
            ax[0].legend()

            ax[1].plot(history.history["loss"], label="Train")
            ax[1].plot(history.history["val_loss"], label="Validation")
            ax[1].set_title("Loss")
            ax[1].legend()

            st.pyplot(fig)
            plt.close(fig)

    # =============== CATS VS DOGS ZIP ===============
    else:
        st.subheader("🐱🐶 Upload Cats vs Dogs ZIP")
        st.info(
            "Best format: separate Cat and Dog folders. "
            "Mixed images are supported only when filenames identify "
            "their class, for example cat.123.jpg and dog.456.jpg."
        )

        zip_file = st.file_uploader(
            "Upload ZIP of dataset",
            type=["zip"],
            key="dataset_zip",
        )

        if zip_file is not None:
            zip_bytes = zip_file.getvalue()
            zip_signature = f"{zip_file.name}:{len(zip_bytes)}"

            # Extract once for this uploaded ZIP.
            if st.session_state.dataset_zip_name != zip_signature:
                old_dir = st.session_state.dataset_temp_dir
                if old_dir:
                    shutil.rmtree(old_dir, ignore_errors=True)

                temp_dir = tempfile.mkdtemp(prefix="cats_dogs_")

                try:
                    with zipfile.ZipFile(
                        zip_file, "r"
                    ) as archive:
                        # Check for unsafe ZIP paths.
                        root = os.path.abspath(temp_dir)

                        for member in archive.infolist():
                            target = os.path.abspath(
                                os.path.join(root, member.filename)
                            )

                            if os.path.commonpath([root, target]) != root:
                                raise ValueError(
                                    "The ZIP contains an unsafe file path."
                                )

                        archive.extractall(temp_dir)

                    st.session_state.dataset_temp_dir = temp_dir
                    st.session_state.dataset_zip_name = zip_signature

                except Exception as exc:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    st.error(f"Could not extract ZIP: {exc}")
                    st.stop()

            temp_dir = st.session_state.dataset_temp_dir

            # First, look for separate class folders.
            data_dir, class_dirs = find_class_directory(temp_dir)

            # If there are no class folders, try clearly labeled filenames.
            if data_dir is None:
                all_images = image_files(temp_dir)

                if not all_images:
                    st.error("No supported image files were found in the ZIP.")
                    st.stop()

                organized_dir = os.path.join(temp_dir, "_organized_classes")
                copied, ambiguous = make_filename_dataset(
                    all_images, organized_dir
                )

                st.write("Images organized from filenames:", copied)
                st.write("Unrecognized filenames skipped:", ambiguous)

                if copied["Cat"] == 0 or copied["Dog"] == 0:
                    st.error(
                        "Could not identify both classes. Your ZIP may "
                        "contain mixed images without cat/dog labels in "
                        "their filenames. Please organize the images into "
                        "separate Cat and Dog folders, then create a new ZIP."
                    )
                    st.stop()

                data_dir = organized_dir
                class_dirs = [
                    os.path.join(data_dir, "Cat"),
                    os.path.join(data_dir, "Dog"),
                ]

            class_names = [Path(folder).name for folder in class_dirs]

            st.write("Dataset directory:", data_dir)
            st.write("Detected classes:", class_names)

            if len(class_names) != 2:
                st.error("Exactly two classes are required.")
                st.stop()

            datagen = ImageDataGenerator(
                rescale=1.0 / 255,
                validation_split=0.2,
            )

            train_gen = datagen.flow_from_directory(
                data_dir,
                target_size=(img_size, img_size),
                batch_size=batch_size,
                class_mode="categorical",
                classes=class_names,
                subset="training",
                shuffle=True,
                seed=42,
            )

            val_gen = datagen.flow_from_directory(
                data_dir,
                target_size=(img_size, img_size),
                batch_size=batch_size,
                class_mode="categorical",
                classes=class_names,
                subset="validation",
                shuffle=False,
                seed=42,
            )

            st.write("Class mapping:", train_gen.class_indices)
            st.write("Training images:", train_gen.samples)
            st.write("Validation images:", val_gen.samples)

            if train_gen.num_classes != 2:
                st.error("The model requires two classes.")
                st.stop()

            if train_gen.samples == 0 or val_gen.samples == 0:
                st.error(
                    "Not enough images for training and validation. "
                    "Add more images or check the ZIP structure."
                )
                st.stop()

            if st.button("🧠 Train CNN on Uploaded Dataset"):
                with st.spinner("Training the cat-vs-dog model..."):
                    model = build_cnn(
                        (img_size, img_size, 3),
                        train_gen.num_classes,
                    )

                    history = model.fit(
                        train_gen,
                        validation_data=val_gen,
                        epochs=epochs,
                    )

                index_to_class = {
                    index: name
                    for name, index in train_gen.class_indices.items()
                }

                st.session_state.model = model
                st.session_state.img_size = img_size
                st.session_state.class_names = [
                    index_to_class[i]
                    for i in range(train_gen.num_classes)
                ]
                st.session_state.dataset_type = "Cats vs Dogs"

                st.success("Cat-vs-dog training complete!")

                fig, ax = plt.subplots(1, 2, figsize=(12, 4))
                ax[0].plot(history.history["accuracy"], label="Train")
                ax[0].plot(history.history["val_accuracy"], label="Validation")
                ax[0].set_title("Accuracy")
                ax[0].legend()

                ax[1].plot(history.history["loss"], label="Train")
                ax[1].plot(history.history["val_loss"], label="Validation")
                ax[1].set_title("Loss")
                ax[1].legend()

                st.pyplot(fig)
                plt.close(fig)


# ---------------- PREDICTION ----------------
elif section == "Predict":
    st.subheader("🔎 Predict Image Class")

    if st.session_state.model is None:
        st.warning("Train a model first in the Train Model section.")
    else:
        st.write("Active model:", st.session_state.dataset_type)
        st.write("Model classes:", st.session_state.class_names)

        uploaded_file = st.file_uploader(
            "Upload an image",
            type=["jpg", "jpeg", "png"],
            key="prediction_image",
        )

        
        if uploaded_file is not None:
            # Display original image
            original_img = Image.open(uploaded_file).convert("RGB")

            st.image(
                original_img,
                caption="Uploaded Original Image",
                width=350
            )

            # Resize a separate copy for CNN prediction
            img = original_img.resize(
                (
                    st.session_state.img_size,
                    st.session_state.img_size
                )
            )

            img_array = img_to_array(img).astype("float32") / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            pred = st.session_state.model.predict(
                img_array, verbose=0
            )[0]

            index = int(np.argmax(pred))
            class_name = st.session_state.class_names[index]
            confidence = float(pred[index]) * 100

            st.write(
                "Prediction scores:",
                {
                    name: f"{float(score) * 100:.2f}%"
                    for name, score in zip(
                        st.session_state.class_names, pred
                    )
                }
            )

            st.success(
                f"Predicted: {class_name} ({confidence:.2f}% confidence)"
            )

            if confidence < 60:
                st.warning(
                    "The model is uncertain. Try more training data "
                    "or a clearer image."
                )
