import openai

from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from dotenv import load_dotenv
from base import Scaffold
import numpy as np

load_dotenv(override=True)


class Descriptor:

    def __init__(self, model="text-embedding-3-small", output_dim=12):
        """
        Initializes the Descriptor class.

        Args:
            model (str): The name of the OpenAI embedding model to use for generating embeddings.
            output_dim (int): The dimensionality of the output embeddings.
        """
        self.client = openai.Client()
        self.model = model
        self.output_dim = output_dim

    def batch_generate(self, scaffolds: list[Scaffold]):
        """
        Generates embeddings for a batch of scaffolds using threading.

        Args:
            scaffolds (list[Scaffold]): A list of scaffold objects for which embeddings will be generated.

        Returns:
            np.ndarray: A NumPy array containing the embeddings for all scaffolds in the batch.
        """
        with ThreadPoolExecutor(max_workers=16) as executor:
            embeddings = list(
                tqdm(
                    executor.map(self.generate, scaffolds),
                    total=len(scaffolds),
                    desc="Generating embeddings",
                )
            )
        return np.array(embeddings)

    def generate(self, scaffold: Scaffold):
        """
        Generates an embedding for a single scaffold.

        Args:
            scaffold (Scaffold): The scaffold object for which the embedding will be generated.

        Returns:
            list[float]: The embedding vector for the given scaffold.
        """
        text = scaffold.scaffold_name + ": " + "\n" + scaffold.scaffold_code
        response = self.client.embeddings.create(
            input=text, model=self.model, dimensions=self.output_dim
        )
        return response.data[0].embedding
