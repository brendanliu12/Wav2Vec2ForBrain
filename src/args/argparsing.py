import argparse
from src.experiments.b2t_gru_w2v_conformer_experiment import (
    B2TGruAndW2VConformerExperiment,
)
from src.experiments.experiment import Experiment
from pydantic import BaseModel
from src.args.base_args import BaseExperimentArgsModel
from typing import Any, Literal, Type, cast
from src.args.yaml_config import YamlConfig
from src.experiments.b2t_gru_w2v_experiment import (
    B2TGruAndW2VExperiment,
)
from src.experiments.b2t_44_sentence_embedding_experiment import (
    B2T44SentenceEmbeddingExperiment,
)

experiments: dict[str, Type[Experiment]] = {
    "b2p2t_gru+w2v": B2TGruAndW2VExperiment,
    "b2p2t_gru+w2v_conformer": B2TGruAndW2VConformerExperiment,
    "b2p2t_44_sentence_embedding": B2T44SentenceEmbeddingExperiment
}


def str_to_bool(value):
    if value.lower() in ["true", "t"]:
        return True
    elif value.lower() in ["false", "f"]:
        return False
    elif value.lower() in ["none", "n"]:
        return None
    else:
        raise argparse.ArgumentTypeError("Invalid boolean value: {}".format(value))


def str_to_list(value):
    import json

    parsed = json.loads(value)
    if not isinstance(parsed, list):
        raise argparse.ArgumentTypeError("Invalid list value: {}".format(value))
    return parsed


def _parser_from_model(parser: argparse.ArgumentParser, model: Type[BaseModel]):
    "Add Pydantic model to an ArgumentParser"
    fields = model.__fields__

    for name, field in fields.items():

        def get_type_args():
            is_literal = getattr(field.annotation, "__origin__", None) is Literal
            is_bool = getattr(field.type_, "__name__", None) == "bool"
            is_list = getattr(field.outer_type_, "__name__", None) == "list"

            if is_literal:
                return {"type": str, "choices": cast(Any, field.annotation).__args__}
            if is_bool:
                return {"type": str_to_bool}
            if is_list:
                return {"type": str_to_list}
            return {"type": field.type_}

        # Collect argument names: field name and alias (if present)
        arg_names = [f"--{name}"]
        if hasattr(field, "alias") and field.alias:
            arg_names.append(f"--{field.alias}")

        parser.add_argument(
            *arg_names,  # Unpack the list of names
            dest=name,  # Use field name as dest for consistency
            default=field.default,
            help=field.field_info.description,
            **get_type_args(),
        )
    return parser


def _create_arg_parser():
    base_parser = argparse.ArgumentParser()
    base_parser = _parser_from_model(base_parser, BaseExperimentArgsModel)
    base_args, _ = base_parser.parse_known_args()

    experiment_model = experiments[base_args.experiment_type].get_args_model()
    parser = argparse.ArgumentParser(
        description="Machine Learning Experiment Configuration"
    )
    parser = _parser_from_model(parser, experiment_model)
    return parser


def get_experiment_from_args() -> Experiment:
    arg_parser = _create_arg_parser()
    args = arg_parser.parse_args()
    yaml_config = YamlConfig()

    experiment = experiments[args.experiment_type](vars(args), yaml_config.config)
    return experiment
