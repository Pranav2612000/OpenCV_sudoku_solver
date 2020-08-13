import tkinter as tk
def toggle_entry():
    global hidden
    if hidden:
        e.grid()
    else:
        e.grid_remove()
    hidden = not hidden

hidden = False
root = tk.Tk()
e = tk.Entry(root)
e.grid(row=0, column=1)
tk.Button(root, text='Toggle entry', command=toggle_entry).grid(row=0, column=0)
root.mainloop()
