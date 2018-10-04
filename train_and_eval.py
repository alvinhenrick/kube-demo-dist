import argparse

import tensorflow as tf

from my_estimator import cnn_model_fn, train_input_fn, eval_input_fn

tf.logging.set_verbosity(tf.logging.DEBUG)


def train_and_evaluate(main_args):
    EVAL_INTERVAL = 300  # seconds
    TRAIN_STEPS = 25000
    EVAL_STEPS = None

    run_config = tf.estimator.RunConfig(save_checkpoints_secs=EVAL_INTERVAL,
                                        keep_checkpoint_max=3)

    # Create a custom estimator using my_model_fn to define the model
    tf.logging.info("Before classifier construction")

    params = {'embedding_initializer': tf.random_uniform_initializer(-1.0, 1.0)}

    estimator = tf.estimator.Estimator(
        model_fn=cnn_model_fn,
        model_dir=main_args.output_path,
        config=run_config,
        params=params)  # Path to where checkpoints etc are stored
    tf.logging.info("...done constructing classifier")

    # Train our model, use the previously function train_input_fn

    tf.logging.info("Before classifier.train")
    train_spec = tf.estimator.TrainSpec(
        input_fn=train_input_fn,
        max_steps=TRAIN_STEPS)

    tf.logging.info("...done classifier.train")

    # Evaluate our model using the eval_input_fn
    # Return value will contain evaluation_metrics such as: loss & average_loss
    tf.logging.info("Before classifier.evaluate")
    eval_spec = tf.estimator.EvalSpec(
        input_fn=eval_input_fn,  # no need to batch in eval
        steps=EVAL_STEPS,
        start_delay_secs=60,  # start evaluating after N seconds
        throttle_secs=EVAL_INTERVAL,  # evaluate every N seconds
    )

    evaluate_result = tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    tf.logging.info("...done classifier.evaluate")
    tf.logging.info("Evaluation results")
    tf.logging.info(evaluate_result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default="")

    args = parser.parse_args()

    train_and_evaluate(args)


if __name__ == '__main__':
    main()
