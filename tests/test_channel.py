# import pytest
#
# from ovgenpy import channel
# from ovgenpy.ovgenpy import default_config
# from ovgenpy.triggers import NullTriggerConfig
#
#
# @pytest.mark.parametrize('wav_prefix', '. nonexistent'.split())
# def test_channel_prefix(wav_prefix, mocker):
#     """ Test that channels are created with the correct wav_prefix. """
#
#     # Create channels, but stub out wave files.
#     mocker.patch.object(channel, 'Wave')
#
#     cfg = default_config(trigger=NullTriggerConfig())
