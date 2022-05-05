import numpy as np
import matplotlib.pyplot as plt

class Show_images(object):
    """
    Scroll through slices. Takes an unspecified number of subfigures per figure.
    suptitles: either a str or a list. Represents the 
    main title of a figure. 
    images_titles: a list with tuples, each tuple an np.array and a 
    title for the array subfigure. 
    """
    def __init__(self, suptitles, *images_titles, colorbar=True, y_label=True):
        # if string if given, make list with that title for 
        # each slice.

        self.colorbar = colorbar
        self.y_label = y_label

        if type(suptitles) == str: 
            self.suptitles = []
            for i in range(images_titles[0][0].shape[2]): 
                self.suptitles.append(suptitles)
        else: 
            self.suptitles = suptitles
                    
        self.fig, self.ax = plt.subplots(1,len(images_titles))

        # split tuples with (image, title) into lists
        self.images = [x[0] for x in images_titles]
        self.titles = [x[1] for x in images_titles]

        # get the number of slices that are to be shown
        rows, cols, self.slices = self.images[0].shape        
        self.ind = 0

        self.fig.suptitle(self.suptitles[self.ind]) # set title 

        self.plots = []
        
        # start at slice 10 if more than 20 slices, 
        # otherwise start at middle slice.
        if self.images[0].shape[2] > 20: 
            self.ind = 10
        else:
            self.ind = self.images[0].shape[2] // 2
        
        # make sure ax is an np array
        if type(self.ax) == np.ndarray:
            pass
        else: 
            self.ax = np.array([self.ax])
        
        # create title for each subfigure in slice
        for (sub_ax, image, title) in zip(self.ax, self.images, self.titles): 
            sub_ax.set_title(title, fontsize=15)

            sub_ax.set_yticklabels([])
            sub_ax.set_xticklabels([])
            
            plot = sub_ax.imshow(image[:, :, self.ind], vmin=image.min(), vmax=image.max())

            self.plots.append(plot)

            if self.colorbar: 
                self.fig.colorbar(plot, ax=sub_ax, fraction=0.046, pad=0.04)

        # link figure to mouse scroll movement
        self.plot_show = self.fig.canvas.mpl_connect('scroll_event', self.onscroll)

        

    def onscroll(self, event):
        """
        Shows next or previous slice with mouse scroll.
        """
        if event.button == 'up':
            self.ind = (self.ind - 1) % self.slices
        else:
            self.ind = (self.ind + 1) % self.slices
        
        self.update()
        

    def update(self):
        """
        Updates the figure.
        """
        self.fig.suptitle(self.suptitles[self.ind])
        
        for i, (plot, image) in enumerate(zip(self.plots, self.images)):
            plot.set_data(image[:, :, self.ind])        

        if self.y_label:
            self.ax[0].set_ylabel('Slice Number: %s' % self.ind)

        
        self.plots[0].axes.figure.canvas.draw()