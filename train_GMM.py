import numpy as np
import argparse

from dataset_plotters import DatasetPlotters
from pose_model_serializer import PoseModelSerializer
from pose_recognition_GMM import PoseRecognitionGMM

# Available covariance types for the GMM
COVARIANCE_TYPE = ['spherical', 'diag', 'tied', 'full']

# Normalisation methods per mode
NO_HIP_MODE_NORMALISATION_METHOD = [
    'No_Normalization',
    'RShoulder_LShoulder__RShoulder_LShoulder',
    'RShoulder_LShoulder__3xRShoulder_LShoulder',
    'Nose_Neck__RShoulder_LShoulder',
    'Nose_Neck__Nose_Neck',
    'Nose_Neck__3xNose_Neck']
    
FULL_MODE_NORMALISATION_METHOD = [
    'No_Normalization',
    'Nose_MidHip__RShoulder_LShoulder',
    'Nose_MidHip__Nose_MidHip',
    'Nose_MidHip__3xNose_MidHip',
    'RShoulder_LShoulder__RShoulder_LShoulder',
    'RShoulder_LShoulder__3xRShoulder_LShoulder']

# Vector dimension 
FULL_MODE_VECTOR_DIM = 16
NO_HIP_MODE_VECTOR_DIM = 14

# Default value for all arguments
FULL_MODE_CMD = 'full_mode'
NOT_HIP_MODE_CMD = 'no_hip_mode'
INPUT_DIR = './dataset/'
OUTPUT_DIR = './models/'
DATASET_MAX_SIZE = 29
FULL_MODE_DIR = 'FULL_MODE_models/'
NO_HIP_MODE_DIR = 'NO_HIP_MODE_models/'

def create_report(output_file, report_content, mode):
    report = open(output_file + 'Report_' + mode + '.txt', 'w')
    report.write('******************** GMM Model Report ********************\n\n')
    for element in report_content:
        report.write('******************** Model ********************' + '\n')
        report.write('Normalization method: ' + 
                     element['normalisation_method'] + '\n')
        report.write('Covariance type: ' +
                     element['covariance_type'] + '\n')
        report.write('train_accuracy: ' +
                     str(element['stats']['train_accuracy']) + '\n')
        report.write('test_accuracy: ' +
                     str(element['stats']['test_accuracy']) + '\n')
        report.write('filename: ' + element['model_path'] + '\n\n')
    report.close()

def train_models(input_dir=INPUT_DIR,
                 output_dir=OUTPUT_DIR,
                 mode=FULL_MODE_CMD,
                 plot_all_data=False,
                 dataset_size=DATASET_MAX_SIZE):

    # Select vector dimension and normalisation methods
    normalization_methods = []
    data_dim = 0
    mode_dir = ''
    if mode is FULL_MODE_CMD:
        normalization_methods = FULL_MODE_NORMALISATION_METHOD
        data_dim = FULL_MODE_VECTOR_DIM
        mode_dir = FULL_MODE_DIR
    else:
        normalization_methods = NO_HIP_MODE_NORMALISATION_METHOD
        data_dim = NO_HIP_MODE_VECTOR_DIM
        mode_dir = NO_HIP_MODE_DIR
    # Config serializer
    dataset_poses_serializer = PoseModelSerializer(dataset_path=input_dir,
                                                   dataset_max_size=dataset_size,
                                                   data_dim=data_dim)
    # Config plotter
    plotter = DatasetPlotters('', #TODO can we eliminate this argument?
                              show_graph=plot_all_data,
                              save_graph=True, #TODO add an argument for this
                              data_dim=data_dim)
    # Obtain process data from openPose
    human_poses_by_normalization = {}
    for method in normalization_methods:
        dataset_poses_serializer.set_normalization_method(method)
        human_poses = dataset_poses_serializer.load_all_poses_from_dataset()
        human_poses_by_normalization[method] = human_poses
    # Plot all for every model
    if plot_all_data:
        plotter.plot_all(human_poses_by_normalization)

    # Train Model
    data_out = []
    # Run every normalisation methods
    for normalization_method, data in human_poses_by_normalization.items():
        # Run every GMM cov type
        for cov in COVARIANCE_TYPE:
            model_path = normalization_method + '_' + cov + '.joblib'
            model = PoseRecognitionGMM(data=data,
                                       covariance=cov, 
                                       save_path=mode_dir + model_path)
            model.train_GMM()
            stats = model.get_model_stats()
            model.save_model()
            data_out.append({'covariance_type': cov,
                             'stats': stats,
                             'normalisation_method':normalization_method,
                             'model_path': model_path})
    # Create final report
    create_report(output_dir, data_out, mode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train GMM')
    parser.add_argument('--mode',
                        type=str,
                        default=FULL_MODE_CMD,
                        help='Choose normalisation method.')
    parser.add_argument('--input-file',
                        type=str,
                        default=INPUT_DIR,
                        help='Dataset input file.')
    parser.add_argument('--output-file',
                        type=str,
                        default=OUTPUT_DIR,
                        help='Dataset output file.')
    #TODO this sould be automatic
    parser.add_argument('--dateset-size',
                        type=int,
                        default=DATASET_MAX_SIZE,
                        help='Dataset size.')
    parser.add_argument('--plot-data',
                        action='store_true',
                        help='Plot data treated.')
    args = parser.parse_args()

    # create dataset from images
    train_models(input_dir=args.input_file,
                 output_dir=args.output_file,
                 mode=args.mode,
                 dataset_size=args.dateset_size,
                 plot_all_data=args.plot_data)
