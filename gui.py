from tkinter import *
from tkinter.scrolledtext import ScrolledText
import tkinter.filedialog
import tensorflow as tf
import struct
from tensorflow.core.example import example_pb2
# import run_summarization


class SimpleGUI(object):
  '''class to create a simple GUI.'''
  def __init__(self):
    self.root = Tk()
    self.root.title("automatic summarization prototype system")
    # Create some frames as container
    self.root.resizable(width=False, height=False)
    self.root.minsize(width=600, height=500)
    self.frame_top = Frame(self.root,width=580, height=305, bg='white')
    
    self.frame_center = Frame(self.root, width=280, height=30)
    self.frame_center_left = Frame(self.frame_center, width=80, height=30)
    self.frame_center_left.pack(side=LEFT)
    self.frame_center_span1 = Frame(self.frame_center, width=40, height=30)
    self.frame_center_span1.pack(side=LEFT)
    self.frame_center_center = Frame(self.frame_center, width=80, height=30)
    self.frame_center_center.pack(side=LEFT)
    self.frame_center_span2 = Frame(self.frame_center, width=40, height=30)
    self.frame_center_span2.pack(side=LEFT)
    self.frame_center_right = Frame(self.frame_center, width=80, height=30)
    self.frame_center_right.pack(side=RIGHT)
    
    self.frame_bottom = Frame(self.root, width=580, height=110, bg='white')
    
    # Create some elements  
    self.source_article = ScrolledText(self.frame_top,width=80, height=23)
    #self.source_article.pack(expand=1, fill="both")
    self.button_open = Button(self.frame_center_left, text="Open", command=self.open)
    self.button_clear = Button(self.frame_center_center, fg='red', text="Clear", command=self.clear)
    self.button_generate = Button(self.frame_center_right, fg='blue', text="Generate", command=self.generate)
    self.summarization = ScrolledText(self.frame_bottom,width=80, height=8)
    #self.summarization.pack(expand=1, fill="both")
    
    # Set each container's position using grid
    self.frame_top.grid(row=0, column=0, padx=2, pady=5)
    self.frame_center.grid(row=1, column=0)
    self.frame_bottom.grid(row=2, column=0, padx=2, pady=5)
    self.frame_top.grid_propagate(0)
    self.frame_center.grid_propagate(0)
    self.frame_bottom.grid_propagate(0)

    # Fill elements into frame
    self.source_article.grid()
    self.summarization.grid()
    self.button_open.grid()
    self.button_clear.grid()
    self.button_generate.grid()
    
  
  def generate(self):
    self.summarization.delete(0.0, END)
    content = self.source_article.get('0.0', END)
    print(content)
    self.write_to_bin('test.bin', content)
  
  def clear(self):
    self.source_article.delete(0.0, END)

  def open(self):
    filename = tkinter.filedialog.askopenfilename()
    if filename != '':
      file_object = open(filename,'r')
      try:
        all_the_text = file_object.read()
        self.source_article.insert(END, all_the_text)
      finally:
        file_object.close()

  def write_to_bin(self, filename, content):
    article = bytes(content, encoding = "utf8")
    abstract =bytes('', encoding = "utf8")
    
    with open(filename, 'wb') as writer:
      # Write to tf.Example
      tf_example = example_pb2.Example()
      tf_example.features.feature['article'].bytes_list.value.extend([article])
      tf_example.features.feature['abstract'].bytes_list.value.extend([abstract])
      tf_example_str = tf_example.SerializeToString()
      str_len = len(tf_example_str)
      writer.write(struct.pack('q', str_len))
      writer.write(struct.pack('%ds' % str_len, tf_example_str))

# def main(self):
    # d = SimpleGUI()
    # sum()
    # mainloop()

# if __name__== "__main__":
    # tf.app.run()