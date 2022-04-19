proj2.py
Running: python3 proj2.py --data_dir <path_to_data_dir> --model_file <path_to_save_model>
    data_dir: defaults to ./flowers/
        directory structure must be:
            <data_dir>/
                <class_name>/
                    <image_name>.jpg
                    ...
                <class_name>/
                    <image_name>.jpg
                    ...
                ...
    model_file: defaults to model.h5.
Requires:
    tensorflow

proj2_test.py
Running: python3 proj2_test.py --model <path_to_saved_model> --test_csv <path_to_test_csv>
    model: defaults to model.h5
    test_csv: defaults to flowers_test.csv
        test_csv must have two columns:
            image_path, label
Requires:
    tensorflow
    pandas

model.h5
    trained model file

