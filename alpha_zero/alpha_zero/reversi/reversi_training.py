from .ai import ReversiDualNetwork
from .reversi_self_match import run_self_match
from .reversi_self_match import load_last_self_match_record
from .reversi_self_match import record_to_model_fitting_data


def train_network():
    record = load_last_self_match_record()
    input, target = record_to_model_fitting_data(record)
    dual_network = ReversiDualNetwork()
    dual_network.fit(input=input, target=target)


def run_training_cycle():
    GAME_COUNT: int = 500
    TEMPERATURE: float = 1.0
    EVALUATION_COUNT: int = 50

    run_self_match(game_count=GAME_COUNT,
                   temperature=TEMPERATURE,
                   evaluation_count=EVALUATION_COUNT)

    train_network()


if __name__ == "__main__":
    for _ in range(10):
        run_training_cycle()
