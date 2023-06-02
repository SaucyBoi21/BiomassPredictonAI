# -*- coding: utf-8 -*-
"""

Create a new function that extracts the distance values of each pixel from a RGBD image.
It presents the mean distance, max distance and min distance besides the frequency and 
a histogram plot.

It has been made based on analyze depth values from PLANTCV
Created on Mon Jan 23 20:39:29 2023

@author: jcard
"""

import os
import numpy as np
from plantcv.plantcv import deprecation_warning, params
from plantcv.plantcv import outputs
from plantcv.plantcv._debug import _debug
from plantcv.plantcv.visualize import histogram
from plotnine import labs

def analyze_depth_values(depth_array, mask, histplot=None, label="default"):
    """This extracts the depth values of each pixel writes the values out to
       a file. It can also print out a histogram plot of pixel intensity
       and a pseudocolor image of the plant.
    Inputs:
    array        = numpy array of depth values
    mask         = Binary mask made from selected contours
    histplot     = if True plots histogram of depth values
    label        = optional label parameter, modifies the variable name of observations recorded
    Returns:
    analysis_image = output image
    :param depth_array: numpy.ndarray
    :param mask: numpy.ndarray
    :param histplot: bool
    :param label: str
    :return analysis_image: ggplot
    """
    if histplot is not None:
        deprecation_warning("'histplot' will be deprecated in a future version of PlantCV. "
                            "This function creates a histogram by default.")

    # Store debug mode
    debug = params.debug

    # apply plant shaped mask to image and calculate statistics based on the masked image
    masked_depth = depth_array[np.where(mask > 0)]
    maxdistance = np.amax(masked_depth)
    mindistance = np.amin(masked_depth)
    avgdistance = np.average(masked_depth) 
    mediandistance = np.median(masked_depth)

    # call the histogram function
    params.debug = None
    hist_fig, hist_data = histogram(depth_array, mask=mask, hist_data=True)
    bin_labels, hist_percent = hist_data['pixel intensity'].tolist(), hist_data['proportion of pixels (%)'].tolist()

    # Store data into outputs class
    outputs.add_observation(sample=label, variable='max_distance', trait='maximum distance',
                            method='plantcv.plantcv.analyze_depth_values', scale='degrees', datatype=float,
                            value=maxdistance, label='degrees')
    outputs.add_observation(sample=label, variable='min_distance', trait='minimum distance',
                            method='plantcv.plantcv.analyze_depth_values', scale='degrees', datatype=float,
                            value=mindistance, label='degrees')
    outputs.add_observation(sample=label, variable='mean_distance', trait='mean distance',
                            method='plantcv.plantcv.analyze_depth_values', scale='degrees', datatype=float,
                            value=avgdistance, label='degrees')
    outputs.add_observation(sample=label, variable='median_distance', trait='median distance',
                            method='plantcv.plantcv.analyze_depth_values', scale='degrees', datatype=float,
                            value=mediandistance, label='degrees')
    outputs.add_observation(sample=label, variable='depth_frequencies', trait='depth frequencies',
                            method='plantcv.plantcv.analyze_depth_values', scale='frequency', datatype=list,
                            value=hist_percent, label=bin_labels)
    # Restore user debug setting
    params.debug = debug

    # change column names of "hist_data"
    hist_fig = hist_fig + labs(x="distance (mm)", y="Proportion of pixels (%)")

    # Print or plot histogram
    _debug(visual=hist_fig, filename=os.path.join(params.debug_outdir, str(params.device) + "_therm_histogram.png"))

    analysis_image = hist_fig
    # Store images
    outputs.images.append(analysis_image)

    return analysis_image

analyze_depth_values(img, mask, histplot=True, label="default")