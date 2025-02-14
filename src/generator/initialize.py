import random
from base import (
    Scaffold,
    Population,
    initialize_session,
)
from prompts.initial_population import (
    COT,
    COT_SC,
    Reflexion,
    LLM_debate,
    Take_a_step_back,
    QD,
    Role_Assignment,
)

# from rich import print
from descriptor import Descriptor
from evals import Validator

from evals import Validator
from descriptor import Clusterer
import asyncio
import logging
import datetime
import copy


def initialize_population_id(args) -> str:
    """
    Initializes the first generation of scaffolds for a given population.

    Args:
        args: Arguments object containing configurations for the population initialization.

    Returns:
        str: The unique ID of the initialized population.
    """
    for session in initialize_session():

        archive = [
            COT,
            COT_SC,
            Reflexion,
            LLM_debate,
            Take_a_step_back,
            QD,
            Role_Assignment,
        ]

        population = Population(session=session, population_benchmark=args.benchmark)
        descriptor = Descriptor()

        validator = Validator(args)
        clusterer = Clusterer()

        generation_timestamp = datetime.datetime.utcnow()

        for scaffold in archive:
            scaffold = Scaffold(
                session=session,
                scaffold_name=scaffold["name"],
                scaffold_code=scaffold["code"],
                scaffold_thought_process=scaffold["thought"],
                population=population,
                generation_timestamp=generation_timestamp,
            )
            population.scaffolds.append(scaffold)

        for scaffold in population.scaffolds:
            scaffold.update(scaffold_descriptor=descriptor.generate(scaffold))

        population_id = str(population.population_id)

        scaffolds_for_validation = (
            session.query(Scaffold)
            .filter_by(population_id=population_id, scaffold_fitness=None)
            .all()
        )

        validator.validate(scaffolds_for_validation)

        # Re-load the population object in this session
        population = (
            session.query(Population).filter_by(population_id=population_id).one()
        )
        # Recluster the population
        clusterer.cluster(population)

    return population_id
