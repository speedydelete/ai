
import discord
import tokenizer
from ai import GPT


ai = GPT(
    vocab_size = 11000,
    maxlen = 1024,
    embed_dim = 512,
    blocks = 8,
    heads = 8,
    ff_dim = 1024,
)
ai.load('87m.keras')
ai.compile()
ai.summary()


RESPONSE_CHANNELS = [
    1342600306164236288, # #speedy-ai-auto
    1342968278531706890, # #general-ai-test-wildeast
]

TRAINING_CHANNELS = [
    1342600306164236288, # #speedy-ai
    1342600306164236288, # #speedy-ai-auto
    1342450286668550228, # #sam-ai
    1342968278531706890, # #general-ai-test-wildeast
]

RESPONSE_AND_TRAINING_CHANNELS = RESPONSE_CHANNELS + TRAINING_CHANNELS

COMMAND_GUILD_IDS = [
    1342450286668550225, # Teaching AI
    1341042144743391286, # Wildsouth
]

COMMAND_GUILDS = list(map(lambda id: discord.Object(id=id), COMMAND_GUILD_IDS))


def run_and_retry(prompt: str, **kwargs) -> str:
    if 'temperature' in kwargs:
        if kwargs['temperature'] > 10:
            return 'temperature cannot be more than 10'
        if kwargs['temperature'] < 0.1:
            return 'temperature cannot be less than 0.1'
    out = ai(prompt, **kwargs)
    attempts = 0
    while out == '' and attempts < 8:
        out = ai(prompt, **kwargs)
        attempts += 1
    if out == '':
        out = 'the ai couldn\'t understand your request'
    return out


intents = discord.Intents.default()
intents.message_content = True # type: ignore
client = discord.Client(intents=intents)
tree = discord.app_commands.CommandTree(client)

@client.event
async def on_ready():
    for guild in COMMAND_GUILDS:
        await tree.sync(guild=guild)
    print(f'logged in as {client.user}')


before_saved = 0

@client.event
async def on_message(message: discord.Message) -> None:
    if message.author.bot:
        return
    if message.channel.id in RESPONSE_AND_TRAINING_CHANNELS:
        prompt = message.content
        if message.reference != None:
            prompt += ' <reply> ' + (await message.channel.fetch_message(message.reference.message_id)).content + ' </reply>' # type: ignore
        if message.channel.id in RESPONSE_CHANNELS:
            await message.reply(run_and_retry(prompt, max_tokens=64))
        if message.channel.id in TRAINING_CHANNELS:
            # ai.train(message.content, 10)
            # global before_saved
            # before_saved += 1
            # if before_saved > 3:
            #     ai.save('87m.keras')
            #     before_saved = 0
            with open('training_data.txt', 'r', encoding='utf-8') as file:
                data = file.read()
            data += '\n\n' + message.content
            with open('training_data.txt', 'w', encoding='utf-8') as file:
                file.write(data)


@tree.command(name='run', description='Sends a message to the AI', guilds=COMMAND_GUILDS)
async def command_run(interaction: discord.Interaction, prompt: str, temperature: float = 1.0) -> None:
    await interaction.response.defer()
    await interaction.followup.send(run_and_retry(prompt, temperature=temperature))

@tree.command(name='tokenize', description='Tokenizes AI content', guilds=COMMAND_GUILDS)
async def command_tokenize(interaction: discord.Interaction, text: str) -> None:
    await interaction.response.send_message(' '.join([str(token) for token in tokenizer.tokenize(text)]))

@tree.command(name='detokenize', description='Tokenizes AI content', guilds=COMMAND_GUILDS)
async def command_detokenize(interaction: discord.Interaction, text: str) -> None:
    try:
        tokens = list(map(int, text.split(' ')))
    except ValueError:
        await interaction.response.send_message('not a space-seperated list of integers')
    else:
        await interaction.response.send_message(tokenizer.detokenize(tokens))

@tree.command(name='count_tokens', description='Finds the number of tokens a message is', guilds=COMMAND_GUILDS)
async def command_token_count(interaction: discord.Interaction, text: str) -> None:
    await interaction.response.send_message(str(len(tokenizer.tokenize(text))))

@tree.command(name='train', description='Trains the AI', guilds=COMMAND_GUILDS)
async def command_train(interaction: discord.Interaction, text: str) -> None:
    ai.train(text)
    await interaction.response.send_message('complete')

@tree.command(name='stop', description='Stops the bot, can only be used by speedydelete', guilds=COMMAND_GUILDS)
async def command_eval(interaction: discord.Interaction, code: str) -> None:
    if interaction.user.id == 1253852708826386518:
        await interaction.response.send_message('shutting down')
        raise SystemExit(0)
    else:
        await interaction.response.send_message('you are not allowed to use /stop')


with open('token.txt', 'r') as file:
    token = file.read()
client.run(token)
