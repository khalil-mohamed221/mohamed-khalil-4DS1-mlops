import argparse
from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    run_pipeline_with_mlflow,
)


def run_prepare():
    print("\n=========== STEP 1: PREPARE DATA ONLY ===========")
    X_train, X_test, y_train, y_test, cat_idx = prepare_data()
    print("\n🎉 Data preparation completed successfully!")
    return X_train, X_test, y_train, y_test, cat_idx


def run_train():
    print("\n=========== STEP 1: PREPARE DATA ===========")
    X_train, X_test, y_train, y_test, cat_idx = prepare_data()

    print("\n=========== STEP 2: TRAIN MODEL ONLY ===========")
    model = train_model(X_train, y_train, cat_idx)

    print("\n=========== STEP 3: SAVE MODEL ===========")
    save_model(model)

    print("\n🎉 Training completed successfully!")


def run_evaluate():
    print("\n=========== STEP 1: PREPARE DATA ===========")
    X_train, X_test, y_train, y_test, cat_idx = prepare_data()

    print("\n=========== STEP 2: TRAIN MODEL ===========")
    model = train_model(X_train, y_train, cat_idx)

    print("\n=========== STEP 3: EVALUATE MODEL ONLY ===========")
    evaluate_model(model, X_test, y_test)

    print("\n🎉 Evaluation completed successfully!")


def run_pipeline():
    print("\n=========== STEP 1: PREPARE DATA ===========")
    X_train, X_test, y_train, y_test, cat_idx = prepare_data()

    print("\n=========== STEP 2: TRAIN MODEL ===========")
    model = train_model(X_train, y_train, cat_idx)

    print("\n=========== STEP 3: EVALUATE MODEL ===========")
    evaluate_model(model, X_test, y_test)

    print("\n=========== STEP 4: SAVE MODEL ===========")
    save_model(model)

    print("\n🎉 Full pipeline executed successfully!")


def main():

    parser = argparse.ArgumentParser(description="ML Pipeline Controller")

    parser.add_argument(
        "--prepare", action="store_true", help="Run only the data preparation step"
    )

    parser.add_argument(
        "--train", action="store_true", help="Run only the training step"
    )

    parser.add_argument(
        "--evaluate", action="store_true", help="Run only the evaluation step"
    )

    parser.add_argument(
        "--pipeline", action="store_true", help="Run the entire ML pipeline"
    )

    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Run the full pipeline with MLflow tracking",
    )

    args = parser.parse_args()

    if args.prepare:
        run_prepare()
    elif args.train:
        run_train()
    elif args.evaluate:
        run_evaluate()
    elif args.pipeline:
        run_pipeline()
    elif args.mlflow:
        run_pipeline_with_mlflow()

    else:
        print("\n❌ No argument provided. Use one of these:")
        print("   --prepare   : prepare data only")
        print("   --train     : train model only")
        print("   --evaluate  : evaluate model only")
        print("   --pipeline  : full pipeline")


if __name__ == "__main__":
    main()
