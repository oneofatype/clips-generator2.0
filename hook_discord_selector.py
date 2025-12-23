"""
Hook Discord Selector
======================

Discord bot that posts generated hooks and allows users to:
1. Select a hook by number (1-8)
2. Provide their own custom hook
3. Auto-fallback to AI evaluation after 20 minutes of no response
"""

import discord
from discord.ext import commands
import os
import asyncio
from dotenv import load_dotenv
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass

# Load environment variables
load_dotenv()


@dataclass
class HookSelectionResult:
    """Result from Discord hook selection."""
    selected_hook: Optional[str]  # The selected or custom hook text
    hook_number: Optional[int]    # 1-8 if selected from list, None if custom
    is_custom: bool               # True if user provided custom hook
    timed_out: bool               # True if no response in 20 minutes


def run_hook_selector_bot(hooks: List[Dict[str, str]], timeout_seconds: int = 1200, transcript: str = "") -> HookSelectionResult:
    """
    Run Discord bot to get user hook selection.
    
    Args:
        hooks: List of dicts with 'number', 'tone', and 'text' keys
        timeout_seconds: How long to wait for response (default 20 minutes)
        transcript: Video transcript text to display for context
        
    Returns:
        HookSelectionResult with selected hook or timeout status
    """
    import discord
    from discord.ext import commands
    import asyncio
    
    # Result container (mutable to capture from async)
    result = {
        'selected_hook': None,
        'hook_number': None,
        'is_custom': False,
        'timed_out': True
    }
    
    # Get credentials from environment
    TARGET_CHANNEL_ID = int(os.getenv('TARGET_CHANNEL_ID'))
    BOT_TOKEN = os.getenv('DISCORD_BOT_TOKEN')
    ALLOWED_USER_IDS = [768854120848293889, 761913055751307284]
    
    # Initialize bot
    intents = discord.Intents.all()
    bot = commands.Bot(command_prefix='!', intents=intents)
    
    # Format transcript message (truncate if too long for Discord)
    transcript_msg = ""
    if transcript:
        # Discord has 2000 char limit per message, leave room for formatting
        truncated_transcript = transcript[:1500] if len(transcript) > 1500 else transcript
        if len(transcript) > 1500:
            truncated_transcript += "... [truncated]"
        transcript_msg = f"📝 **VIDEO TRANSCRIPT:**\n```\n{truncated_transcript}\n```\n\n"
    
    # Format hooks message
    hooks_message = "🎣 **HOOK SELECTION TIME!**\n\n"
    hooks_message += "Here are the generated hooks. Reply with:\n"
    hooks_message += "• A number (1-8) to select that hook\n"
    hooks_message += "• Your own custom hook text\n"
    hooks_message += f"• Or wait {timeout_seconds // 60} minutes for AI auto-selection\n\n"
    
    # Build hooks list without code block (to avoid truncation issues)
    hooks_list = ""
    for hook in hooks:
        hooks_list += f"**{hook['number']}.** [{hook['tone']}]: \"{hook['text']}\"\n"
    
    hooks_message += hooks_list + "\n"
    hooks_message += f"<@768854120848293889> <@761913055751307284>"
    
    @bot.event
    async def on_ready():
        print(f'[DISCORD] Hook selector bot ready! Logged in as {bot.user}')
        channel = bot.get_channel(TARGET_CHANNEL_ID)
        if channel:
            # Send transcript first if available (separate message due to length)
            if transcript_msg:
                await channel.send(transcript_msg)
            # Then send hooks
            await channel.send(hooks_message)
            print(f'[DISCORD] Hooks posted to channel')
        else:
            print(f'[DISCORD] Could not find channel with ID {TARGET_CHANNEL_ID}')
        
        # Start timeout countdown
        asyncio.create_task(auto_timeout())
    
    @bot.event
    async def on_message(message):
        nonlocal result
        
        # Ignore bot's own messages
        if message.author == bot.user:
            return
        
        # Only process messages in target channel from allowed users
        if message.channel.id != TARGET_CHANNEL_ID:
            return
        
        if message.author.id not in ALLOWED_USER_IDS:
            return
        
        content = message.content.strip()
        
        # Check if it's a number selection (1-8)
        if content.isdigit():
            num = int(content)
            if 1 <= num <= len(hooks):
                selected_hook = hooks[num - 1]['text']
                result['selected_hook'] = selected_hook
                result['hook_number'] = num
                result['is_custom'] = False
                result['timed_out'] = False
                
                await message.add_reaction('✅')
                await message.channel.send(f"✅ **Hook #{num} selected!**\n```{selected_hook}```\nProceeding with this hook...")
                
                # Close bot after short delay
                await asyncio.sleep(2)
                await bot.close()
                return
            else:
                await message.add_reaction('❌')
                await message.channel.send(f"❌ Invalid number. Please choose 1-{len(hooks)}")
                return
        
        # Check if it's a custom hook (more than 3 words, less than 100 chars)
        words = content.split()
        if len(words) >= 3 and len(content) <= 100:
            result['selected_hook'] = content
            result['hook_number'] = None
            result['is_custom'] = True
            result['timed_out'] = False
            
            await message.add_reaction('✅')
            await message.channel.send(f"✅ **Custom hook accepted!**\n```{content}```\nProceeding with your hook...")
            
            # Close bot after short delay
            await asyncio.sleep(2)
            await bot.close()
            return
        
        await bot.process_commands(message)
    
    @bot.command()
    async def skip(ctx):
        """Skip Discord selection and use AI evaluation."""
        nonlocal result
        if ctx.author.id not in ALLOWED_USER_IDS:
            return
        
        result['timed_out'] = True  # Triggers AI fallback
        await ctx.send("⏭️ Skipping to AI evaluation...")
        await asyncio.sleep(1)
        await bot.close()
    
    @bot.command()
    async def shutdown(ctx):
        """Force shutdown the bot."""
        if ctx.author.id not in ALLOWED_USER_IDS:
            await ctx.send("❌ Not authorized")
            return
        await ctx.send("📴 Shutting down...")
        await bot.close()
    
    async def auto_timeout():
        """Auto-close after timeout and trigger AI fallback."""
        # Countdown warnings
        warning_times = [
            (timeout_seconds - 300, "⏰ 5 minutes remaining to select a hook!"),
            (timeout_seconds - 60, "⏰ 1 minute remaining!"),
            (timeout_seconds - 10, "⏰ 10 seconds remaining...")
        ]
        
        elapsed = 0
        channel = bot.get_channel(TARGET_CHANNEL_ID)
        
        for wait_until, warning_msg in warning_times:
            if wait_until > elapsed:
                await asyncio.sleep(wait_until - elapsed)
                elapsed = wait_until
                if channel and result['selected_hook'] is None:
                    await channel.send(warning_msg)
        
        # Wait remaining time
        remaining = timeout_seconds - elapsed
        if remaining > 0:
            await asyncio.sleep(remaining)
        
        # Check if still no selection
        if result['selected_hook'] is None:
            if channel:
                await channel.send("⏰ **Time's up!** No selection made. Using AI to evaluate and select the best hook...")
            await asyncio.sleep(2)
            await bot.close()
    
    # Run the bot
    try:
        bot.run(BOT_TOKEN)
    except Exception as e:
        print(f"[DISCORD] Bot error: {e}")
    
    # Return result
    return HookSelectionResult(
        selected_hook=result['selected_hook'],
        hook_number=result['hook_number'],
        is_custom=result['is_custom'],
        timed_out=result['timed_out']
    )


def parse_hooks_from_generator_output(raw_output: str) -> List[Dict[str, str]]:
    """
    Parse the hook generator's output into a list of hooks.
    
    Args:
        raw_output: Raw text output from the generator agent
        
    Returns:
        List of dicts with 'number', 'tone', and 'text' keys
    """
    import re
    
    hooks = []
    
    # Pattern to match: 1. [TONE]: "Hook text here"
    # Uses a more robust pattern that captures everything between quotes
    # Handles: 1. [TONE]: "text with 'inner quotes' and *special* chars"
    pattern = r'(\d+)\.\s*\[([^\]]+)\][:\-\s]*"([^"]*(?:"[^"]*)*)"'
    
    matches = re.findall(pattern, raw_output)
    
    # If the pattern didn't work well, try alternative patterns
    if len(matches) < 4:
        # Try line-by-line parsing for more robustness
        lines = raw_output.strip().split('\n')
        hooks = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Match: 1. [TONE]: "text" or 1. [TONE]: text
            match = re.match(r'^(\d+)\.\s*\[([^\]]+)\][:\-\s]*"?(.+?)"?\s*$', line)
            if match:
                number, tone, text = match.groups()
                # Clean up the text - remove trailing quote if present
                text = text.strip().rstrip('"').strip()
                if text:  # Only add if there's actual text
                    hooks.append({
                        'number': int(number),
                        'tone': tone.strip().upper(),
                        'text': text
                    })
    else:
        for match in matches:
            number, tone, text = match
            hooks.append({
                'number': int(number),
                'tone': tone.strip().upper(),
                'text': text.strip()
            })
    
    # Sort by number
    hooks.sort(key=lambda x: x['number'])
    
    return hooks


# =============================================================================
# MUSIC SELECTOR
# =============================================================================

@dataclass
class MusicSelectionResult:
    """Result from Discord music selection."""
    music_path: Optional[str]  # Path to the music file (downloaded or default)
    is_custom: bool            # True if user provided custom music
    timed_out: bool            # True if no response in timeout period
    source_url: Optional[str]  # Original URL if downloaded


def run_music_selector_bot(default_music_path: str = "bg_music.mp3", 
                            timeout_seconds: int = 1200,
                            download_folder: str = "temp") -> MusicSelectionResult:
    """
    Run Discord bot to get user music selection.
    
    Args:
        default_music_path: Path to default background music
        timeout_seconds: How long to wait for response (default 20 minutes)
        download_folder: Folder to save downloaded music
        
    Returns:
        MusicSelectionResult with music path or default
    """
    import discord
    from discord.ext import commands
    import asyncio
    
    # Result container
    result = {
        'music_path': default_music_path,
        'is_custom': False,
        'timed_out': True,
        'source_url': None
    }
    
    # Get credentials from environment
    TARGET_CHANNEL_ID = int(os.getenv('TARGET_CHANNEL_ID'))
    BOT_TOKEN = os.getenv('DISCORD_BOT_TOKEN')
    ALLOWED_USER_IDS = [768854120848293889, 761913055751307284]
    
    # Initialize bot
    intents = discord.Intents.all()
    bot = commands.Bot(command_prefix='!', intents=intents)
    
    # Format music selection message
    music_message = "🎵 **BACKGROUND MUSIC SELECTION**\n\n"
    music_message += "Would you like to add custom background music?\n\n"
    music_message += "**Options:**\n"
    music_message += "• Send a **YouTube URL** to download and use that music\n"
    music_message += "• Type **`default`** to use the default background music\n"
    music_message += "• Attach an **audio file** (.mp3, .wav, .m4a)\n"
    music_message += f"• Or wait {timeout_seconds // 60} minutes to auto-use default\n\n"
    music_message += f"Default music: `{default_music_path}`\n\n"
    music_message += f"<@768854120848293889> <@761913055751307284>"
    
    @bot.event
    async def on_ready():
        print(f'[DISCORD] Music selector bot ready! Logged in as {bot.user}')
        channel = bot.get_channel(TARGET_CHANNEL_ID)
        if channel:
            await channel.send(music_message)
            print(f'[DISCORD] Music selection prompt posted')
        else:
            print(f'[DISCORD] Could not find channel with ID {TARGET_CHANNEL_ID}')
        
        # Start timeout countdown
        asyncio.create_task(auto_timeout())
    
    @bot.event
    async def on_message(message):
        nonlocal result
        
        # Ignore bot's own messages
        if message.author == bot.user:
            return
        
        # Only process messages in target channel from allowed users
        if message.channel.id != TARGET_CHANNEL_ID:
            return
        
        if message.author.id not in ALLOWED_USER_IDS:
            return
        
        content = message.content.strip().lower()
        
        # Check for "default" command
        if content == "default" or content == "skip":
            result['music_path'] = default_music_path
            result['is_custom'] = False
            result['timed_out'] = False
            
            await message.add_reaction('✅')
            await message.channel.send(f"✅ **Using default music:** `{default_music_path}`")
            
            await asyncio.sleep(2)
            await bot.close()
            return
        
        # Check for YouTube URL
        if "youtube.com" in content or "youtu.be" in content:
            await message.add_reaction('⏳')
            await message.channel.send("⏳ Downloading music from YouTube...")
            
            try:
                downloaded_path = await download_youtube_audio(message.content.strip(), download_folder)
                if downloaded_path:
                    result['music_path'] = downloaded_path
                    result['is_custom'] = True
                    result['timed_out'] = False
                    result['source_url'] = message.content.strip()
                    
                    await message.add_reaction('✅')
                    await message.channel.send(f"✅ **Music downloaded!** Saved to: `{downloaded_path}`")
                    
                    await asyncio.sleep(2)
                    await bot.close()
                    return
                else:
                    await message.add_reaction('❌')
                    await message.channel.send("❌ Failed to download. Please try again or type `default`")
            except Exception as e:
                await message.add_reaction('❌')
                await message.channel.send(f"❌ Download error: {str(e)[:100]}")
            return
        
        # Check for audio file attachment
        if message.attachments:
            for attachment in message.attachments:
                if attachment.filename.lower().endswith(('.mp3', '.wav', '.m4a', '.ogg', '.aac')):
                    await message.add_reaction('⏳')
                    await message.channel.send("⏳ Downloading attached audio...")
                    
                    try:
                        # Save attachment
                        save_path = os.path.join(download_folder, f"custom_music_{attachment.filename}")
                        await attachment.save(save_path)
                        
                        result['music_path'] = save_path
                        result['is_custom'] = True
                        result['timed_out'] = False
                        
                        await message.add_reaction('✅')
                        await message.channel.send(f"✅ **Audio saved!** Using: `{save_path}`")
                        
                        await asyncio.sleep(2)
                        await bot.close()
                        return
                    except Exception as e:
                        await message.add_reaction('❌')
                        await message.channel.send(f"❌ Save error: {str(e)[:100]}")
                    return
        
        await bot.process_commands(message)
    
    async def auto_timeout():
        """Auto-close after timeout and use default music."""
        # Countdown warnings
        warning_times = [
            (timeout_seconds - 300, "⏰ 5 minutes remaining to select music!"),
            (timeout_seconds - 60, "⏰ 1 minute remaining!"),
            (timeout_seconds - 10, "⏰ 10 seconds remaining...")
        ]
        
        elapsed = 0
        channel = bot.get_channel(TARGET_CHANNEL_ID)
        
        for wait_until, warning_msg in warning_times:
            if wait_until > elapsed:
                await asyncio.sleep(wait_until - elapsed)
                elapsed = wait_until
                if channel and result['music_path'] == default_music_path and result['timed_out']:
                    await channel.send(warning_msg)
        
        # Wait remaining time
        remaining = timeout_seconds - elapsed
        if remaining > 0:
            await asyncio.sleep(remaining)
        
        # Check if still no selection
        if result['timed_out']:
            if channel:
                await channel.send(f"⏰ **Time's up!** Using default music: `{default_music_path}`")
            await asyncio.sleep(2)
            await bot.close()
    
    # Run the bot
    try:
        bot.run(BOT_TOKEN)
    except Exception as e:
        print(f"[DISCORD] Music bot error: {e}")
    
    # Return result
    return MusicSelectionResult(
        music_path=result['music_path'],
        is_custom=result['is_custom'],
        timed_out=result['timed_out'],
        source_url=result['source_url']
    )


async def download_youtube_audio(url: str, output_folder: str) -> Optional[str]:
    """
    Download audio from YouTube URL using yt-dlp.
    
    Args:
        url: YouTube video URL
        output_folder: Folder to save the audio
        
    Returns:
        Path to downloaded audio file, or None if failed
    """
    import subprocess
    import glob
    
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    output_template = os.path.join(output_folder, "yt_music_%(id)s.%(ext)s")
    
    try:
        # Use yt-dlp to download audio
        cmd = [
            "yt-dlp",
            "-x",  # Extract audio
            "--audio-format", "mp3",
            "--audio-quality", "192K",
            "-o", output_template,
            "--no-playlist",
            url
        ]
        
        # Run in executor to not block async
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: subprocess.run(cmd, check=True, capture_output=True))
        
        # Find the downloaded file
        pattern = os.path.join(output_folder, "yt_music_*.mp3")
        files = glob.glob(pattern)
        
        if files:
            # Return most recently created file
            return max(files, key=os.path.getctime)
        
        return None
        
    except Exception as e:
        print(f"[MUSIC] YouTube download error: {e}")
        return None


# =============================================================================
# STANDALONE TEST
# =============================================================================

if __name__ == "__main__":
    # Test with sample hooks
    test_hooks = [
        {'number': 1, 'tone': 'SARCASTIC', 'text': 'Oh sure, trading your life for a pension is "safe"'},
        {'number': 2, 'tone': 'DISMISSIVE', 'text': 'Nobody cares about your retirement plan'},
        {'number': 3, 'tone': 'TEASING', 'text': 'The real risk nobody talks about...'},
        {'number': 4, 'tone': 'CONTRARIAN', 'text': 'Entrepreneurs are the safe ones actually'},
        {'number': 5, 'tone': 'SKEPTICAL', 'text': 'Is a 40-year job really less risky?'},
        {'number': 6, 'tone': 'FAUX-NAIVE', 'text': 'Wait, pensions can get cut?'},
        {'number': 7, 'tone': 'SARCASTIC', 'text': 'Yes, definitely trust your pension fund'},
        {'number': 8, 'tone': 'TEASING', 'text': 'What they never tell you about retirement'},
    ]
    
    print("Testing Hook Discord Selector...")
    print("=" * 60)
    
    result = run_hook_selector_bot(test_hooks, timeout_seconds=120)  # 2 min for testing
    
    print("\n" + "=" * 60)
    print("RESULT:")
    print(f"  Selected Hook: {result.selected_hook}")
    print(f"  Hook Number: {result.hook_number}")
    print(f"  Is Custom: {result.is_custom}")
    print(f"  Timed Out: {result.timed_out}")
