from gnuradio import blocks
from gnuradio import channels
from gnuradio import eng_notation
from gnuradio import gr
from gnuradio.eng_option import eng_option
from gnuradio.filter import firdes
from optparse import OptionParser
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--sps', type=int,default=2)
parser.add_argument('--samp_rate', type=int,default=32000)
parser.add_argument('--source', type=str,default='/home/gou/np')
parser.add_argument('--out', type=str,default='/home/gou/np1')
parser.add_argument('--pn_mag', type=int,default=-100)
parser.add_argument('--iq_mag_imb', type=float,default=0.8)
parser.add_argument('--iq_ph_imb', type=float,default=0.1)
parser.add_argument('--quad_offset', type=float,default=0.1)
parser.add_argument('--inp_offset', type=float,default=0.1)
parser.add_argument('--freq_offset', type=float,default=-0.2)

data = parser.parse_args()

system = gr.top_block()
sps = data.sps
samp_rate = data.samp_rate
channels_impairments = channels.impairments(data.pn_mag, data.iq_mag_imb, data.iq_ph_imb, data.quad_offset, data.inp_offset, data.freq_offset, 0, 0)
channels_selective_fading_model = channels.selective_fading_model( 8, 10.0/samp_rate, False, 4.0, 0, (0.1,0.1,1.3), (1,0.99,0.97), 8)
blocks_throttle = blocks.throttle(gr.sizeof_gr_complex*1, samp_rate,True)
blocks_file_source = blocks.file_source(gr.sizeof_gr_complex*1, data.source, False)
blocks_file_sink = blocks.file_sink(gr.sizeof_gr_complex*1, data.out, False)
blocks_file_sink.set_unbuffered(False)


system.connect((blocks_file_source, 0), (blocks_throttle, 0))
system.connect((blocks_throttle, 0), (channels_impairments, 0))
system.connect((channels_impairments, 0), (channels_selective_fading_model, 0))
system.connect((channels_selective_fading_model, 0), (blocks_file_sink, 0))

system.run()
