import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from lime_explanation import apply_perturbation_to_signal, perturb_mean  

def plot_class_distribution(labels, title="Class Distribution"):
    """
    Plots the distribution of classes using a bar chart, with specific colors for each class.
    
    Parameters:
    - labels (pd.Series): A pandas Series containing class labels.
    - title (str): Title for the plot.
    """
    # Define specific colors for each class
    class_colors = {1: "r", 2: "g", 3: "b", 4: "k"}
    
    # Set the aesthetic style of the plots
    sns.set_style("whitegrid")

    plt.figure(figsize=(6, 4))
    ax = sns.countplot(x=labels)
    ax.set_title(title)
    
    # Get unique classes and their counts
    class_counts = labels.value_counts().sort_index()

    # Iterate over the unique classes and set the colors for each bar
    for i, class_id in enumerate(class_counts.index):
        ax.patches[i].set_color(class_colors[class_id])

    plt.xlabel("Class")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45)  # Rotate class labels to avoid overlap, if necessary
    plt.show()



def plot_sample_signals(features, labels):
    """
    Plots one sample signal from each class in the dataset.
    
    Parameters:
        features (DataFrame): The features of the ECG dataset, where each row is a signal.
        labels (Series): The labels for the dataset, indicating the class of each signal.
    """
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 4))

    unique_classes = labels.unique()
    for class_ in unique_classes:
        sample_index = labels[labels == class_].index[0]
        if class_ == 1:
            plt.plot(features.loc[sample_index, :], label=f"Class {class_}", color="r")
        elif class_ == 2:
            plt.plot(features.loc[sample_index, :], label=f"Class {class_}", color="g")
        elif class_ == 3:
            plt.plot(features.loc[sample_index, :], label=f"Class {class_}", color="b")
        elif class_ == 4:
            plt.plot(features.loc[sample_index, :], label=f"Class {class_}", color="k")

    plt.title("Sample Signal from Each Class")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend(title="Classes")
    plt.show()

def plot_segmented_signal(instance_sig, slice_width, title):
    """
    Plots the signal and its segments.

    Parameters:
        instance_sig (np.ndarray): The signal instance to plot.
        slice_width (int): The width of each slice in the segmented signal.
        title (str): Title for the plot.
    """
    plt.figure(figsize=(12, 3))
    plt.plot(instance_sig, label='Signal')
    num_slices = len(instance_sig) // slice_width
    
    for i in range(1, num_slices):
        plt.axvline(x=i * slice_width, color='r', linestyle='--')

    plt.title(title)
    plt.xlabel('Time Index')
    plt.ylabel('Signal Amplitude')
    plt.legend()
    plt.show()

def plot_perturbed_signal(original_sig, perturbed_sig, perturbation, num_slices, title='ECG Signal with Perturbation'):
    """
    Plots the original and perturbed  signals with slices and deactivated segments highlighted for each channel.

    Parameters:
    - original_sig (np.ndarray): The original  signal, should be a 2D array with shape (num_samples, num_channels).
    - perturbed_sig (np.ndarray): The perturbed  signal after applying the perturbation, should be a 2D array with shape (num_samples, num_channels).
    - perturbation (np.ndarray): The perturbation vector used to modify the  signal, should be a 1D array with length num_slices.
    - num_slices (int): The total number of segments the  signal is divided into.
    - title (str): The title for the plot. Optional.
    """
    num_channels = original_sig.shape[1]
    total_length = original_sig.shape[0]
    slice_width = total_length // num_slices

    plt.figure(figsize=(15, 4 * num_channels))
    channel_labels = ['FHR', 'TOCO', 'Gest Age']

    for ch in range(num_channels):
        # Plot original  signal with slices and deactivated segments highlighted
        plt.subplot(num_channels, 2, 2 * ch + 1)
        plt.plot(original_sig[:, ch], label='Original Signal', color='black')
        plt.title(f'Original Signal - {channel_labels[ch]}')
        for i in range(num_slices):
            start_idx = i * slice_width
            end_idx = min((i + 1) * slice_width, total_length)
            plt.axvline(x=start_idx, color='r', linestyle='--', alpha=0.5)  # Slice boundary
            if perturbation[i] == 0:  # If the segment is "off" in the perturbation
                plt.axvspan(start_idx, end_idx, color='red', alpha=0.3)  # Highlight deactivated segment
        plt.xlabel('Time')
        plt.ylabel('Amplitude')

        # Plot perturbed  signal with slices and deactivated segments highlighted
        plt.subplot(num_channels, 2, 2 * ch + 2)
        plt.plot(perturbed_sig[:, ch], label='Perturbed Signal', color='green')
        plt.title(f'Perturbed {title} - {channel_labels[ch]}')
        for i in range(num_slices):
            start_idx = i * slice_width
            end_idx = min((i + 1) * slice_width, total_length)
            plt.axvline(x=start_idx, color='black', linestyle='--', alpha=0.5)  # Slice boundary
            if perturbation[i] == 0:  # If the segment is "off" in the perturbation
                plt.axvspan(start_idx, end_idx, color='yellow', alpha=0.3)  # Highlight deactivated segment
        plt.xlabel('Time')
        plt.ylabel('Amplitude')

    plt.tight_layout()
    plt.show()


def visualize_lime_explanation_(instance_sig, top_influential_segments, num_slices):
    """
    Visualizes the original signal and highlights the top influential segments
    identified by a LIME explanation.

    Parameters:
    - instance_sig (np.ndarray): The original signal.
    - top_influential_segments (np.ndarray): Indices of the top influential segments.
    - num_slices (int): The number of segments the signal is divided into.
    """
    plt.figure(figsize=(15, 12))

    channel_labels = ['FHR', 'TOCO', 'Gest Age']

    for i in range(3):
        # Plot the original signal for each channel with highlighted segments
        plt.subplot(3, 1, i + 1)
        for j in range(1, num_slices):
            plt.axvline(x=j * (len(instance_sig) // num_slices), color='r', linestyle='--')
        plt.plot(instance_sig[:, i], label=f'Original {channel_labels[i]}', color='blue')
        for segment in top_influential_segments:
            start_idx = segment * (len(instance_sig) // num_slices)
            end_idx = start_idx + (len(instance_sig) // num_slices)
            plt.axvspan(start_idx, end_idx, color='yellow', alpha=0.3)  # Highlight influential segments
        plt.title(f' {channel_labels[i]} with Highlighted Segments')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')

    plt.tight_layout()
    plt.show()

def visualize_lime_explanation_color_(instance_sig, top_influential_segments, num_slices, perturb_function=perturb_mean):
    """
    Visualizes the original signal and highlights the top influential segments
    identified by a LIME explanation.

    Parameters:
    - instance_sig (np.ndarray): The original signal.
    - top_influential_segments (np.ndarray): Indices of the top influential segments.
    - num_slices (int): The number of segments the signal is divided into.
    - perturb_function (function): The perturbation function used (default is perturb_mean).
    """
    plt.figure(figsize=(15, 12))

    channel_labels = ['FHR', 'TOCO', 'Gest Age']

    for i in range(3):
        # Plot the original signal for each channel with highlighted segments
        plt.subplot(3, 1, i + 1)
        for j in range(1, num_slices):
            plt.axvline(x=j * (len(instance_sig) // num_slices), color='r', linestyle='--')
        plt.plot(instance_sig[:, i], label=f'Original {channel_labels[i]}', color='blue')
        
        for segment in top_influential_segments:
            start_idx = segment * (len(instance_sig) // num_slices)
            end_idx = start_idx + (len(instance_sig) // num_slices)
            importance_coefficient = instance_sig[start_idx:end_idx, i].mean() / instance_sig[:, i].std()
            alpha = min(abs(importance_coefficient * 0.6), 1.0)  # Clip alpha to be within 0-1 range
            if importance_coefficient > 0:
                plt.axvspan(start_idx, end_idx, color='red', alpha=alpha)  # Positive influence which push results to pathological
            else:
                plt.axvspan(start_idx, end_idx, color='green', alpha=alpha)  # Negative influence which push results to healty
        
        plt.title(f'{channel_labels[i]} with Highlighted Segments')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')

    plt.tight_layout()
    plt.show()