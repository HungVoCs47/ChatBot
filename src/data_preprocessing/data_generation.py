import openai
import os
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import (ChatPromptTemplate,
                               PromptTemplate,
                               SystemMessagePromptTemplate,
                               AIMessagePromptTemplate,
                               HumanMessagePromptTemplate,
                               )

from pydantic import BaseModel, Field, NonNegativeInt
from typing import List, Optional, Type
from langchain.output_parsers import PydanticOutputParser


ENDC = '\33[0m'
RED = '\33[31m'
GREEN = '\33[32m'
YELLOW = '\33[33m'
VIOLET = '\33[35m'
CYAN = '\33[36m'

load_dotenv()


openai_api_key = os.getenv('OPENAI_API_KEY')
#print(openai_api_key)


instruct_llm = OpenAI(model="gpt-3.5-turbo-instruct", 
             openai_api_key=os.getenv('OPENAI_API_KEY'),
             temperature=0, 
             max_tokens=3500)

chat_llm = ChatOpenAI(model="gpt-3.5-turbo", 
             openai_api_key=os.getenv('OPENAI_API_KEY'),
             temperature=0, 
             max_tokens=3500)

import chromadb
persist_directory = './data/vectordb'
client = chromadb.PersistentClient(path=persist_directory)

embeddings_model = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY'))


listing_examples = ['''Headline: A 3-bedroom, 2-bathroom home in Green Oaks
Neighborhood: Green Oaks
Price: €1,200,000
Bedrooms: 3
Bathrooms: 2
House Size (sqm): 110

Description: Welcome to this eco-friendly oasis nestled in the heart of Green Oaks. This charming 3-bedroom, 2-bathroom home boasts energy-efficient features such as solar panels and a well-insulated structure. Natural light floods the living spaces, highlighting the beautiful hardwood floors and eco-conscious finishes. The open-concept kitchen and dining area lead to a spacious backyard with a vegetable garden, perfect for the eco-conscious family. Embrace sustainable living without compromising on style in this Green Oaks gem.

Neighborhood Description: Green Oaks is a close-knit, environmentally-conscious community with access to organic grocery stores, community gardens, and bike paths. Take a stroll through the nearby Green Oaks Park or grab a cup of coffee at the cozy Green Bean Cafe. With easy access to public transportation and bike lanes, commuting is a breeze.''',
'''Headline: Charming Oasis in Neuilly
Neighborhood: Neuilly
Price: €2,000,000
Bedrooms: 5
Bathrooms: 2
House Size (sqm): 180

Description: Discover this enchanting family home nestled in the heart of Neuilly. With its timeless elegance and modern amenities, it’s the perfect sanctuary for your family.

Neighborhood Description: Neuilly offers a serene lifestyle, top-rated schools, and picturesque parks. Enjoy the tranquility while being conveniently close to amenities.
''']
# convert examples to one string
listing_samples = '\n---------------------------------\n'.join(listing_examples)

# Pydantic models for the listings data
class listings_description_model(BaseModel):
    listings_headline: str = Field(description='property headline')
    listings_neighborhood: str = Field(description='property neighborhood where the home is located')
    listings_price: NonNegativeInt = Field(description='property price')
    listings_bedrooms: NonNegativeInt = Field(description='property number of bedrooms')
    listings_bathrooms: NonNegativeInt = Field(description='property number of bathrooms')
    listings_house_size: NonNegativeInt = Field(description='property house size')
    listings_description: str = Field(description='property description following the guidelines')
    listings_neighborhood_description: str = Field(description='property neighborhood description')

class listings_descriptions(listings_description_model):
    list_of_listings: List[listings_description_model] = Field(description='list of property descriptions following the guidelines')

parser = PydanticOutputParser(pydantic_object=listings_descriptions)
print(parser.get_format_instructions())


# Prompt to generate synthetic listing data using examples and pydantic model
template_genlistings = '''You are a real estate agent tasked with creating property listings for a variety of homes in different neighborhoods. Your goal is to craft compelling descriptions that entice potential buyers. Follow these guidelines for each listing:
1. Headline: Begin with an attention-grabbing opening sentence that captures the viewer's interest in the home.
2. In Bullet points:
    - Neighborhood: The name of the neighborhood where the home is located.
    - **Price**: Mention the price.
    - **Bedrooms**: Specify the number of bedrooms (e.g., 5).
    - **Bathrooms**: Specify the number of bathrooms (e.g., 2).
    - **House Size in sqm**: Provide the exact square footage (e.g., 150).
3. Property Description: Include a brief description of the home and its features.
4. Neighborhood description: Briefly describe the neighborhood.

========== EXEMPLES ==========
{listing_samples}
========== END OF EXAMPLES ==========

Remember to be succinct, informative, and persuasive.
**Only use the requested format.**
{format_instructions}
'''

prompt = PromptTemplate.from_template(template_genlistings,
                                      partial_variables={'format_instructions':parser.get_format_instructions})

system_prompt = prompt.format(listing_samples=listing_samples)

response = openai.chat.completions.create(
          model="gpt-3.0-turbo",
          messages=[
          {
            "role": "system",
            "content": system_prompt
          },
          {
            "role": "user",
            "content": "Generate 15 listings following these guidelines."
          }
          ],
        temperature=0,
        max_tokens=4000,
        )

print(response.choices[0].message['content'][:-2])
