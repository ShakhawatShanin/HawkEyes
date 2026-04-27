import cv2 as cv
import numpy as np
from PIL import Image, ImageTk, ImageGrab
import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess
import platform
import io


class PerspectiveApp:
    def __init__(self):
        self.points = []
        self.img = None
        self.original_img = None

        self.root = tk.Tk()
        self.root.title("Perspective Transformer Tool")

        self.canvas = tk.Canvas(self.root, width=600, height=400, bg='gray')
        self.canvas.pack()

        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)

        upload_btn = tk.Button(button_frame, text="Upload Image", command=self.upload_image)
        upload_btn.grid(row=0, column=0, padx=10)

        paste_btn = tk.Button(button_frame, text="Paste Image", command=self.paste_image)
        paste_btn.grid(row=0, column=1, padx=10)

        reset_btn = tk.Button(button_frame, text="Reset Points", command=self.reset_points)
        reset_btn.grid(row=0, column=2, padx=10)

        save_btn = tk.Button(button_frame, text="Save Image", command=self.save_image)
        save_btn.grid(row=0, column=3, padx=10)

        copy_btn = tk.Button(button_frame, text="Copy Image", command=self.copy_image)
        copy_btn.grid(row=0, column=4, padx=10)

        self.canvas.bind("<Button-1>", self.get_points)

        self.root.mainloop()

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if file_path:
            self.img = cv.imread(file_path)
            self.original_img = self.img.copy()
            self.display_image()
            self.points = []

    def paste_image(self):
        if platform.system() == "Linux":
            try:
                # Paste image from clipboard using xclip or wl-paste
                result = subprocess.run("xclip -selection clipboard -t image/png -o", 
                                         shell=True, stdout=subprocess.PIPE)
                img_data = result.stdout
                if img_data:
                    img_array = np.frombuffer(img_data, np.uint8)
                    self.img = cv.imdecode(img_array, cv.IMREAD_COLOR)
                    self.original_img = self.img.copy()
                    self.display_image()
                else:
                    messagebox.showerror("Error", "No image found on clipboard!")
            except Exception as e:
                messagebox.showerror("Error", f"Clipboard paste failed: {e}")
        else:
            messagebox.showwarning("Warning", "Paste feature only supported on Linux with xclip!")

    def display_image(self):
        if self.img is not None:
            img_rgb = cv.cvtColor(self.img, cv.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_pil = img_pil.resize((600, 400), Image.LANCZOS)

            self.tk_img = ImageTk.PhotoImage(img_pil)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)

    def get_points(self, event):
        if self.img is None:
            messagebox.showerror("Error", "Please upload or paste an image first!")
            return

        if len(self.points) < 4:
            x, y = int(event.x * (self.original_img.shape[1] / 600)), int(event.y * (self.original_img.shape[0] / 400))
            self.points.append([x, y])
            cv.circle(self.img, (x, y), 5, (255, 0, 0), -1)
            self.display_image()

            if len(self.points) == 4:
                self.get_perspective()

    def get_perspective(self):
        width = int(np.linalg.norm(np.array(self.points[1]) - np.array(self.points[0])))
        height = int(np.linalg.norm(np.array(self.points[2]) - np.array(self.points[0])))
        pts1 = np.float32(self.points)
        pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
        matrix = cv.getPerspectiveTransform(pts1, pts2)
        warped_img = cv.warpPerspective(self.original_img, matrix, (width, height))
        self.img = warped_img
        self.display_image()

    def reset_points(self):
        if self.original_img is not None:
            self.img = self.original_img.copy()
            self.points = []
            self.display_image()
        else:
            messagebox.showwarning("Warning", "No image to reset!")

    def save_image(self):
        if self.img is not None:
            save_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")])
            if save_path:
                cv.imwrite(save_path, self.img)
                messagebox.showinfo("Success", "Image saved successfully!")
        else:
            messagebox.showerror("Error", "No image to save!")

    def copy_image(self):
        if self.img is not None:
            img_rgb = cv.cvtColor(self.img, cv.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)

            buffer = io.BytesIO()
            img_pil.save(buffer, format="PNG")

            if platform.system() == "Linux":
                try:
                    # Using wl-copy for Wayland
                    subprocess.run("wl-copy", input=buffer.getvalue(), check=True)
                    messagebox.showinfo("Success", "Image copied!")
                except Exception:
                    # Fallback for xclip (X11)
                    subprocess.run("xclip -selection clipboard -t image/png", input=buffer.getvalue(), shell=True)
                    messagebox.showinfo("Success", "Image copied!")
            else:
                messagebox.showwarning("Warning", "Clipboard copy supported only on Linux with wl-copy or xclip!")
        else:
            messagebox.showerror("Error", "No image to copy!")


if __name__ == "__main__":
    PerspectiveApp()
