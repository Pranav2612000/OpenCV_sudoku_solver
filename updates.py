from tkinter import *
def update_digit(update_display):
    update_display.destroy()
    print("Upda")
def main():
    update_display = Tk()
    update_display.title("Updates")
    update_display.resizable(0,0)
    infolabel = Label(update_display, text = "Enter the cell to be changed(Starting from 1)", font = ("Times New Roman", 25)).grid(columnspan = 3, sticky = W, ipadx = 10, ipady = 10)
    xlabel = Label(update_display, text = "Row Number(Starting from 1):", font = ("Helvetica", 15)).grid(row = 1,column = 0, columnspan = 2, sticky = W, ipadx = 5, ipady = 5)
    x = Entry(update_display).grid(row = 1, column = 2)
    #heading = Label(master, text = "SUDOKU SOLVER", font = ("Roboto", 20)).grid(columnspan = 6, pady = 50, sticky = N)
    ylabel = Label(update_display, text = "Column Number(Starting from 1):", font = ("Helvetica", 15)).grid(row = 2, column = 0, columnspan = 2, sticky = W, ipadx = 5, ipady = 5)
    y = Entry(update_display).grid(row = 2, column = 2)
    Numlabel = Label(update_display, text = "Corrected Digit(0 for blank):", font = ("Helvetica", 15)).grid(row = 3, column = 0, columnspan = 2, sticky = W, ipadx = 5, ipady = 10)
    Num = Entry(update_display).grid(row = 3, column = 2)
    update = Button(update_display, text = "Update", fg = "white", bg = "black", command=lambda: update_digit(update_display)).grid(row = 4, columnspan = 3)
    update_display.mainloop()
if __name__== "__main__":
  main()
