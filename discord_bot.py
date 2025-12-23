def run_discord_bot(shower_thought):
    import discord
    from discord.ext import commands
    import os
    import requests
    from dotenv import load_dotenv
    import re
    from savegif import save_gif
    import asyncio

    # Load environment variables from .env file
    load_dotenv()

    # Initialize bot with command prefix
    bot = commands.Bot(command_prefix='!', intents=discord.Intents.all())

    # Get channel ID and bot token from environment variables
    TARGET_CHANNEL_ID = int(os.getenv('TARGET_CHANNEL_ID'))
    BOT_TOKEN = os.getenv('DISCORD_BOT_TOKEN')

    @bot.event
    async def on_ready():
        print(f'Bot is ready! Logged in as {bot.user}')
        # Get the target channel
        channel = bot.get_channel(TARGET_CHANNEL_ID)
        if channel:
            await channel.send(f"🧠 New Shower Thought:\n```{shower_thought}```\n<@768854120848293889> and <@761913055751307284>\n"
                "Please upload a relevant `.gif`. If none received in 20 minutes, I'll auto-select one.")
        else:
            print(f"Could not find channel with ID {TARGET_CHANNEL_ID}")

        asyncio.create_task(auto_shut_down())

    @bot.event
    async def on_message(message):
        if message.author == bot.user:
            return

        if message.channel.id == TARGET_CHANNEL_ID:
            if message.content.endswith('.gif') or 'giphy.com' in message.content or 'tenor.com' in message.content:
                await message.add_reaction('👍')
                save_gif(message.content.strip())
                await message.channel.send("GIF saved successfully as gif.gif!")

        await bot.process_commands(message)

    @bot.command()
    async def shutdown(ctx):
        allowed_ids = [768854120848293889, 761913055751307284]
        if ctx.author.id not in allowed_ids:
            with open('nikal-lawdey-pehli-fursat.gif', 'rb') as f:
                picture = discord.File(f)
            await ctx.send(file=picture)
        else:
            await ctx.send("Shutting down... 📴")
            await bot.close()

    async def auto_shut_down():
        await asyncio.sleep(1197)
        channel = bot.get_channel(TARGET_CHANNEL_ID)
        if channel:
            await channel.send("I am going to shut down myself in...")
            message = await channel.send("3 ...")
        await asyncio.sleep(1)
        await message.edit(content="2 ..")
        await asyncio.sleep(1)
        await message.edit(content="1 .")
        await asyncio.sleep(1)
        await message.edit(content="Bye 🫡")
        await bot.close()

    bot.run(BOT_TOKEN) 

# run_discord_bot("How to get abs in one day?")