import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import load_model

class BlueButton(ttk.Button):
    def __init__(self, master=None, **kwargs):
        ttk.Button.__init__(self, master, style="Blue.TButton", **kwargs)


model_path = 'my_modelCNN2.h5'
loaded_model = load_model(model_path)

# GUI Class
class CarTypePredictionApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Vehicle Type Prediction")
        self.master.geometry("600x500")
        self.master.configure(bg="#333333")  # Dark gray background

        #custom style
        self.style = ttk.Style()
        self.style.configure("Blue.TButton", padding=(5, 5), font=("Helvetica", 12), background="#2196F3", foreground="#2196F3" ,width=13,
                                 height=32 )

        self.label = tk.Label(master, text="Vehicle classification ", font=("Helvetica", 16), pady=10, bg="#333333", fg="#2196F3")
        self.label.pack(side="top", anchor="center")


        self.upload_button = BlueButton(master, text="Upload Image", command=self.upload_image)
        self.upload_button.pack(anchor='w', pady=(0, 10), padx=10)

        self.predict_button = BlueButton(master, text="Predict", command=self.predict_type)
        self.predict_button.pack(anchor='w', pady=(0, 10),padx=10)

        self.clear_button = BlueButton(master, text="Clear", command=self.clear)
        self.clear_button.pack(anchor='w', pady=(0, 10),padx=10)

        self.result_label = tk.Label(master, text="", font=("Helvetica", 14), pady=10, bg="#333333", fg="white")
        self.result_label.pack()

        self.image_path = None
        self.img_label = tk.Label(master, bg="#333333")
        self.img_label.pack (side="top", pady=(0, 10) ,padx=10)

        self.img_ref = None

    def upload_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if self.image_path:
            img = Image.open(self.image_path)
            img = img.resize((300, 300), Image.BICUBIC)
            img_tk = ImageTk.PhotoImage(img)
            self.img_label.config(image=img_tk)
            self.img_label.image = img_tk
            self.result_label.config(text="")
            self.img_ref = img_tk

    def clear(self):
        self.image_path = None
        self.img_label.config(image="")
        self.result_label.config(text="")
        self.img_ref = None

    def preprocess_image(self, image_path):
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return img_array

    def predict_type(self):
        if self.image_path:
            img_array = self.preprocess_image(self.image_path)
            predictions = loaded_model.predict(img_array)
            predicted_class_index = np.argmax(predictions)

            class_labels = ['Hatchback', 'SUV', 'Sedan', 'truck']
            predicted_class = class_labels[predicted_class_index]

            self.result_label.config(text=f"Predicted vehicle Type: {predicted_class}", fg="#2196F3")
        else:
            self.result_label.config(text="Please upload an image first", fg="#f44336")

def main():
    root = tk.Tk()
    app = CarTypePredictionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
