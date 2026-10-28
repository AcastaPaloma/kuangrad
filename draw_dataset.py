import json
import os
import tkinter as tk

GRID = 16
CELL = 25
DATASET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset.json")


def flatten(grid):
    # Row-major: 16x16 -> 256 values, i.e. a 256 x 1 column vector
    return [float(v) for row in grid for v in row]


def load_dataset(path=DATASET):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def save_sample(grid, label, path=DATASET):
    data = load_dataset(path)
    data.append({"x": flatten(grid), "y": label})
    with open(path, "w") as f:
        json.dump(data, f)
    return len(data)


class App:
    def __init__(self, root):
        self.root = root
        root.title("16x16 Drawer")

        self.grid_state = [[0] * GRID for _ in range(GRID)]
        self.rects = [[None] * GRID for _ in range(GRID)]

        self.canvas = tk.Canvas(root, width=GRID * CELL, height=GRID * CELL,
                                bg="white", highlightthickness=1)
        self.canvas.grid(row=0, column=0, rowspan=4, padx=10, pady=10)
        for r in range(GRID):
            for c in range(GRID):
                x0, y0 = c * CELL, r * CELL
                self.rects[r][c] = self.canvas.create_rectangle(
                    x0, y0, x0 + CELL, y0 + CELL, fill="white", outline="#ccc")

        self.canvas.bind("<Button-1>", self.paint)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-2>", self.erase)
        self.canvas.bind("<B2-Motion>", self.erase)
        self.canvas.bind("<Button-3>", self.erase)
        self.canvas.bind("<B3-Motion>", self.erase)

        tk.Label(root, text="Ground truth label:").grid(row=0, column=1, sticky="w", padx=10)
        self.label_entry = tk.Entry(root, width=10)
        self.label_entry.grid(row=1, column=1, sticky="w", padx=10)

        buttons = tk.Frame(root)
        buttons.grid(row=2, column=1, sticky="w", padx=10, pady=5)
        tk.Button(buttons, text="Save", command=self.save).pack(side="left")
        tk.Button(buttons, text="Clear", command=self.clear).pack(side="left", padx=5)

        self.status = tk.Label(root, text="", fg="#333", anchor="w", justify="left")
        self.status.grid(row=3, column=1, sticky="w", padx=10)

        tk.Label(root, text="Flattened (256 x 1):").grid(row=4, column=0, columnspan=2, sticky="w", padx=10)
        vector_frame = tk.Frame(root)
        vector_frame.grid(row=5, column=0, columnspan=2, sticky="w", padx=10, pady=(0, 10))
        self.vector = tk.Text(vector_frame, width=6, height=12, font=("Courier", 10))
        scrollbar = tk.Scrollbar(vector_frame, command=self.vector.yview)
        self.vector.config(yscrollcommand=scrollbar.set)
        self.vector.pack(side="left")
        scrollbar.pack(side="left", fill="y")

        self.refresh()
        self.update_status(f"Samples saved: {len(load_dataset())}")

    def paint(self, event):
        self.set_cell(event, 1)

    def erase(self, event):
        self.set_cell(event, 0)

    def set_cell(self, event, value):
        c, r = event.x // CELL, event.y // CELL
        if 0 <= r < GRID and 0 <= c < GRID and self.grid_state[r][c] != value:
            self.grid_state[r][c] = value
            self.canvas.itemconfig(self.rects[r][c], fill="black" if value else "white")
            self.refresh()

    def refresh(self):
        flat = flatten(self.grid_state)
        self.vector.config(state="normal")
        self.vector.delete("1.0", "end")
        self.vector.insert("1.0", "\n".join(str(int(v)) for v in flat))
        self.vector.config(state="disabled")

    def clear(self):
        for r in range(GRID):
            for c in range(GRID):
                self.grid_state[r][c] = 0
                self.canvas.itemconfig(self.rects[r][c], fill="white")
        self.refresh()
        self.update_status("Cleared.")

    def save(self):
        text = self.label_entry.get().strip()
        try:
            label = int(text)
        except ValueError:
            self.update_status("Enter an integer label first.")
            return
        total = save_sample(self.grid_state, label)
        self.clear()
        self.update_status(f"Saved label {label}. Samples saved: {total}")

    def update_status(self, message):
        self.status.config(text=message)


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
