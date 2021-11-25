"""
Control the Light Crafter 6500DLP evaluation module over USB. The code is based around the dlp6500 class, which builds
the command packets to be sent to the DMD. However, the details of sending the USB packets is implemented in
operating system specific subclasses which handle the details of sending these packets. Currently, only support
for Windows has been written and tested in the dlp6500win() class.

Extensions to Linux can be accomplished by implementing only two functions, _send_raw_packet() and _get_device(),
in the dlp6500ix() class. This would likely also require importing an Linux compatible HID module.

Although Texas Instruments has an SDK for this evaluation module (http://www.ti.com/tool/DLP-ALC-LIGHTCRAFTER-SDK), but
it is not very well documented and we had difficulty building it. Further, it is intended to produce a static library
which cannot be used with e.g. python and the ctypes library as a dll could be.

This DMD control code was originally based on refactoring in https://github.com/mazurenko/Lightcrafter6500DMDControl.
The combine_patterns() function was inspired by https://github.com/csi-dcsc/Pycrafter6500.
"""
import pywinusb.hid as pyhid
import time
import struct
import numpy as np
import copy


##############################################
# compress DMD pattern data
##############################################
def combine_patterns(patterns, bit_depth=1):
    """
    Given a series of binary patterns, combine these into RGB images to send to DMD.

    For binary patterns, DMD supports sending a group of up to 24 patterns as an RGB image, with each bit of the 24 bit
    RGB values giving the pattern for one image.

    :param patterns: nimgs x ny x nx array of uint8
    :param bit_depth: 1
    :return:
    """

    # todo: don't know if there is a pattern combination for other bit depths?
    if bit_depth != 1:
        raise NotImplementedError('not implemented')

    if not np.all(np.logical_or(patterns == 0, patterns == 1)):
        raise ValueError('patterns must be binary')

    combined_patterns = []

    # determine number of compressed images and create them
    n_combined_patterns = int(np.ceil(len(patterns) / 24))
    for num_pat in range(n_combined_patterns):

        combined_pattern_current = np.zeros((3, patterns.shape[1], patterns.shape[2]), dtype=np.uint8)

        for ii in range(np.min([24, len(patterns) - 24*num_pat])):
            # first 8 patterns encoded in B byte of color image, next 8 in G, last 8 in R
            if ii < 8:
                combined_pattern_current[2, :, :] += patterns[ii + 24*num_pat, :, :] * 2**ii
            elif ii >= 8 and ii < 16:
                combined_pattern_current[1, :, :] += patterns[ii + 24*num_pat, :, :] * 2**(ii-8)
            elif ii >= 16 and ii < 24:
                combined_pattern_current[0, :, :] += patterns[ii + 24*num_pat, :, :] * 2**(ii-16)

        combined_patterns.append(combined_pattern_current)

    return combined_patterns


def split_combined_patterns(combined_patterns):
    """
    Split binary patterns which have been combined into a single uint8 RGB image back to separate images.

    :param combined_patterns: 3 x Ny x Nx uint8 array representing up to 24 combined patterns. Actually
    will accept input of arbitrary dimensions as long as first dimension has size 3.
    :return: 24 x Ny x Nx array. This will always have first dimension of 24 because the number of
    zero patterns at the end is ambiguous.
    """
    patterns = np.zeros((24,) + combined_patterns.shape[1:], dtype=np.uint8)

    for ii in range(8):
        patterns[ii] = (combined_patterns[2] & 2**ii) >> ii

    for ii in range(8, 16):
        patterns[ii] = (combined_patterns[1] & 2 ** (ii-8)) >> (ii-8)

    for ii in range(16, 24):
        patterns[ii] = (combined_patterns[0] & 2 ** (ii - 16)) >> (ii+8)

    return patterns


def encode_erle(pattern):
    """

    'erle': enhanced run length encoding. Similar to RLE, but now the number of repeats byte is given by
    either one or two bytes.

    specification:
    ctrl byte 1, ctrl byte 2, ctrl byte 3, description
    0          , 0          , n/a        , end of image
    0          , 1          , n          , copy n pixels from the same position on the previous line
    0          , n>1        , n/a        , n uncompressed RGB pixels follow
    n>1        , n/a        , n/a        , repeat following pixel n times

    :param pattern: uint8 3 x Ny x Nx array of RGB values, or Ny x Nx array
    :return:
    """

    # pattern must be uint8
    if pattern.dtype != np.uint8:
        raise ValueError('pattern must be of type uint8')

    # if 2D pattern, expand this to RGB with pattern in B layer and RG=0
    if pattern.ndim == 2:
        pattern = np.concatenate((np.zeros((1,) + pattern.shape, dtype=np.uint8),
                                  np.zeros((1,) + pattern.shape, dtype=np.uint8),
                                  np.array(pattern[None, :, :], copy=True)), axis=0)

    if pattern.ndim != 3 and pattern.shape[0] != 3:
        raise ValueError("Image data is wrong shape. Must be 3 x ny x nx, with RGB values in each layer.")

    pattern_compressed = []
    _, ny, nx = pattern.shape

    # todo: not sure if this is allowed to cross row_rgb boundaries? If so, could pattern.ravel() instead of looping
    # todo: don't think above suggestion works, but if last n pixels of above row_rgb are same as first n of this one
    # todo: then with ERLE encoding I can use \x00\x01 Hex(n). But checking this may not be so easy. Right now
    # todo: only implemented if entire rows are the same!
    # todo: erle and rle are different enough probably should split apart more
    # loop over pattern rows
    for ii in range(pattern.shape[1]):
        row_rgb = pattern[:, ii, :]

        # if this row_rgb is the same as the last row_rgb, can communicate this by sending length of row_rgb
        # and then \x00\x01 (copy n pixels from previous line)
        # todo: can also do this for shorter sequences than the entire row_rgb
        if ii > 0 and np.array_equal(row_rgb, pattern[:, ii - 1, :]):
            msb, lsb = erle_len2bytes(nx)
            pattern_compressed += [0x00, 0x01, msb, lsb]
        else:

            # find points along row where pixel value changes
            # for RGB image, change happens when ANY pixel value changes
            value_changed = np.sum(np.abs(np.diff(row_rgb, axis=1)), axis=0) != 0
            # also need to include zero, as this will need to be encoded.
            # add one to index to get position of first new value instead of last old value
            inds_change = np.concatenate((np.array([0]), np.where(value_changed)[0] + 1))

            # get lengths for each repeat, including last one which extends until end of the line
            run_lens = np.concatenate((np.array(inds_change[1:] - inds_change[:-1]),
                                       np.array([nx - inds_change[-1]])))

            # now build compressed list
            for ii, rlen in zip(inds_change, run_lens):
                v = row_rgb[:, ii]
                length_bytes = erle_len2bytes(rlen)
                pattern_compressed += length_bytes + [v[0], v[1], v[2]]

            # for ii in range(len(run_lens)):
            #     v = vals[:, ii]
            #     length_bytes = erle_len2bytes(run_lens[ii])
            #     pattern_compressed += length_bytes + [v[0], v[1], v[2]]

    # bytes indicating image end
    pattern_compressed += [0x00, 0x01, 0x00]

    return pattern_compressed


def encode_rle(pattern):
    """
    Compress pattern use run length encoding (RLE)

    'rle': row_rgb length encoding (RLE). Information is encoded as number of repeats
    of a given value and values. In RLE the number of repeats is given by a single byte.
    e.g. AAABBCCCCD = 3A2B4C1D
    The DMD uses a `24bit RGB' encoding scheme, meaning four bits represent each piece of information. The first byte
    (i.e. the control byte) gives the length, and the next three give the values for RGB.
    The only exceptions occur when the control byte is 0x00, in this case there are several options. If the next byte
    is 0x00 this indicates 'end of line', if it is 0x01 this indicates 'end of image', and if it is any other number n,
    then this indicates the following 3*n bytes are uncompressed
    i.e. \x00 \x03 \xAB\xCD\xEF \x11\x22\x33 \x44\x55\x66 -> \xAB\xCD\xEF \x11\x22\x33 \x44\x55\x66

    specification:
    ctrl byte 1, color byte, description
    0          , 0         , end of line
    0          , 1         , end of image (required)
    0          , n>=2      , n uncompressed RGB pixels follow
    n>0        , n/a       , repeat following RGB pixel n times

    :param pattern:
    :return:
    """
    # pattern must be uint8
    if pattern.dtype != np.uint8:
        raise ValueError('pattern must be of type uint8')

    # if 2D pattern, expand this to RGB with pattern in B layer and RG=0
    if pattern.ndim == 2:
        pattern = np.concatenate((np.zeros((1,) + pattern.shape, dtype=np.uint8),
                                  np.zeros((1,) + pattern.shape, dtype=np.uint8),
                                  np.array(pattern[None, :, :], copy=True)), axis=0)

    if pattern.ndim != 3 and pattern.shape[0] != 3:
        raise ValueError("Image data is wrong shape. Must be 3 x ny x nx, with RGB values in each layer.")

    pattern_compressed = []
    _, ny, nx = pattern.shape

    # loop over pattern rows
    for ii in range(pattern.shape[1]):
        row_rgb = pattern[:, ii, :]

        # if this row_rgb is the same as the last row_rgb, can communicate this by sending length of row_rgb
        # and then \x00\x01 (copy n pixels from previous line)
        # todo: can also do this for shorter sequences than the entire row_rgb
        if ii > 0 and np.array_equal(row_rgb, pattern[:, ii - 1, :]):
            msb, lsb = erle_len2bytes(nx)
            pattern_compressed += [0x00, 0x01, msb, lsb]
        else:

            # find points along row where pixel value changes
            # for RGB image, change happens when ANY pixel value changes
            value_changed = np.sum(np.abs(np.diff(row_rgb, axis=1)), axis=0) != 0
            # also need to include zero, as this will need to be encoded.
            # add one to index to get position of first new value instead of last old value
            inds_change = np.concatenate((np.array([0]), np.where(value_changed)[0] + 1))

            # get lengths for each repeat, including last one which extends until end of the line
            run_lens = np.concatenate((np.array(inds_change[1:] - inds_change[:-1]),
                                       np.array([nx - inds_change[-1]])))

            # now build compressed list
            for ii, rlen in zip(inds_change, run_lens):
                v = row_rgb[:, ii]
                if rlen <= 255:
                    pattern_compressed += [rlen, v[0], v[1], v[2]]
                else:  # if run is longer than one byte, need to break it up

                    counter = 0
                    while counter < rlen:
                        end_pt = np.min([counter + 255, rlen]) - 1
                        current_len = end_pt - counter + 1
                        pattern_compressed += [current_len, v[0], v[1], v[2]]

                        counter = end_pt + 1
            # todo: do I need an end of line character?

    # todo: is this correct for RLE?
    # bytes indicating image end
    pattern_compressed += [0x00]

    return pattern_compressed


def decode_erle(dmd_size, pattern_bytes):
    """
    Decode pattern from ERLE or RLE.

    :param dmd_size: [ny, nx]
    :param pattern_bytes: list of bytes representing encoded pattern
    :return:
    """

    ii = 0  # counter tracking position in compressed byte array
    line_no = 0  # counter tracking line number
    line_pos = 0  # counter tracking next position to write in line
    current_line = np.zeros((3, dmd_size[1]), dtype=np.uint8)
    rgb_pattern = np.zeros((3, 0, dmd_size[1]), dtype=np.uint8)
    # todo: maybe should rewrite popping everything to avoid dealing with at least one counter?
    while ii < len(pattern_bytes):

        # reset each new line
        if line_pos == dmd_size[1]:
            rgb_pattern = np.concatenate((rgb_pattern, current_line[:, None, :]), axis=1)
            current_line = np.zeros((3, dmd_size[1]), dtype=np.uint8)
            line_pos = 0
            line_no += 1
        elif line_pos >= dmd_size[1]:
            raise ValueError("While reading line %d, length of line exceeded expected value" % line_no)

        # end of image denoted by single 0x00 byte
        if ii == len(pattern_bytes) - 1:
            if pattern_bytes[ii] == 0:
                break
            else:
                raise ValueError('Image not terminated with 0x00')

        # control byte of zero indicates special response
        if pattern_bytes[ii] == 0:

            # end of line
            if pattern_bytes[ii + 1] == 0:
                ii += 1
                continue

            # copy bytes from previous lines
            elif pattern_bytes[ii + 1] == 1:
                if pattern_bytes[ii + 2] < 128:
                    n_to_copy = pattern_bytes[ii + 2]
                    ii += 3
                else:
                    n_to_copy = erle_bytes2len(pattern_bytes[ii + 2:ii + 4])
                    ii += 4

                # copy bytes from same position in previous line
                current_line[:, line_pos:line_pos + n_to_copy] = rgb_pattern[:, line_no-1, line_pos:line_pos + n_to_copy]
                line_pos += n_to_copy

            # next n bytes unencoded
            else:
                if pattern_bytes[ii + 1] < 128:
                    n_unencoded = pattern_bytes[ii + 1]
                    ii += 2
                else:
                    n_unencoded = erle_bytes2len(pattern_bytes[ii + 1:ii + 3])
                    ii += 3

                for jj in range(n_unencoded):
                    current_line[0, line_pos + jj] = int(pattern_bytes[ii + 3*jj])
                    current_line[1, line_pos + jj] = int(pattern_bytes[ii + 3*jj + 1])
                    current_line[2, line_pos + jj] = int(pattern_bytes[ii + 3*jj + 2])

                ii += 3 * n_unencoded
                line_pos += n_unencoded

            continue

        # control byte != 0, regular decoding
        # get block len
        if pattern_bytes[ii] < 128:
            block_len = pattern_bytes[ii]
            ii += 1
        else:
            block_len = erle_bytes2len(pattern_bytes[ii:ii + 2])
            ii += 2

        # write values to lists for rgb colors
        current_line[0, line_pos:line_pos + block_len] = np.asarray([pattern_bytes[ii]] * block_len, dtype=np.uint8)
        current_line[1, line_pos:line_pos + block_len] = np.asarray([pattern_bytes[ii + 1]] * block_len, dtype=np.uint8)
        current_line[2, line_pos:line_pos + block_len] = np.asarray([pattern_bytes[ii + 2]] * block_len, dtype=np.uint8)
        ii += 3
        line_pos += block_len

    return rgb_pattern


def erle_len2bytes(length):
    """
    Encode a length between 0-2**15-1 as 1 or 2 bytes for use in erle encoding format.

    Do this in the following way: if length < 128, encode as one byte
    If length > 128, then encode as two bits. Create the least significant byte (LSB) as follows: set the most
    significant bit as 1 (this is a flag indicating two bytes are being used), then use the least signifcant 7 bits
    from length. Construct the most significant byte (MSB) by throwing away the 7 bits already encoded in the LSB.

    i.e.
    lsb = (length & 0x7F) | 0x80
    msb = length >> 7


    :param length: integer 0-(2**15-1)
    :return:
    """

    # check input
    if isinstance(length, float):
        if length.is_integer():
            length = int(length)
        else:
            raise TypeError('length must be convertible to integer.')

    # if not isinstance(length, int):
    #     raise Exception('length must be an integer')

    if length < 0 or length > 2 ** 15 - 1:
        raise ValueError('length is negative or too large to be encoded.')

    # main function
    if length < 128:
        len_bytes = [length]
    else:
        # i.e. lsb is formed by taking the 7 least significant bits and extending to 8 bits by adding
        # a 1 in the msb position
        lsb = (length & 0x7F) | 0x80
        # second byte obtained by throwing away first 7 bits and keeping what remains
        msb = length >> 7
        len_bytes = [lsb, msb]

    return len_bytes


def erle_bytes2len(byte_list):
    """
    Convert a 1 or 2 byte list in little endian order to length
    :param byte_list: [byte] or [lsb, msb]
    :return:
    """
    # if msb is None:
    #     length = lsb
    # else:
    #     length = (msb << 7) + (lsb - 0x80)
    if len(byte_list) == 1:
        length = byte_list[0]
    else:
        lsb, msb = byte_list
        length = (msb << 7) + (lsb - 0x80)

    return length


##############################################
# controlling dmd
##############################################
class dlp6500:
    """
    Base class for communicating with DLP6500
    """

    # tried to match with the DLP6500 GUI names where possible
    command_dict = {'PAT_START_STOP': 0x1A24,
                    'DISP_MODE': 0x1A1B,
                    'MBOX_DATA': 0x1A34,
                    'PAT_CONFIG': 0x1A31,
                    'PATMEM_LOAD_INIT_MASTER': 0x1A2A,
                    'PATMEM_LOAD_DATA_MASTER': 0x1A2B,
                    'TRIG_OUT1_CTL': 0x1A1D,
                    'TRIG_OUT2_CTL': 0x1A1E,
                    'TRIG_IN1_CTL': 0x1A35,
                    'TRIG_IN2_CTL': 0x1A36}

    err_dictionary = {'no error': 0,
                      'batch file checksum error': 1,
                      'device failure': 2,
                      'invalid command number': 3,
                      'incompatible controller/dmd': 4,
                      'command not allowed in current mode': 5,
                      'invalid command parameter': 6,
                      'item referred by the parameter is not present': 7,
                      'out of resource (RAM/flash)': 8,
                      'invalid BMP compression type': 9,
                      'pattern bit number out of range': 10,
                      'pattern BMP not present in flash': 11,
                      'pattern dark time is out of range': 12,
                      'signal delay parameter is out of range': 13,
                      'pattern exposure time is out of range': 14,
                      'pattern number is out of range': 15,
                      'invalid pattern definition': 16,
                      'pattern image memory address is out of range': 17,
                      'internal error': 255}

    def __init__(self, vendor_id=0x0451, product_id=0xc900, debug=True):
        """
        Get instance of DLP LightCrafter evaluation module (DLP6500 or DLP9000). This is the base class which os
        dependent classes should inherit from. The derived classes only need to implement _get_device and
        _send_raw_packet.

        :param vendor_id: vendor id, used to find DMD USB device
        :param product_id: product id, used to find DMD USB device
        :param bool debug: If True, will print output of commands.
        """

        # USB packet length not including report_id_byte
        self.packet_length_bytes = 64
        self.debug = debug

        # find device
        self._get_device(vendor_id, product_id)

    def __del__(self):
        pass

    # sending and receiving commands, operating system dependence
    def _get_device(self, vendor_id, product_id):
        """
        Return handle to DMD. This command can contain OS dependent implementation

        :param vendor_id: usb vendor id
        :param product_id: usb product id
        :return:
        """
        pass

    def _send_raw_packet(self, buffer, listen_for_reply=False, timeout=5):
        """
        Send a single USB packet.

        todo: this command can contain os dependent implementation
        one interesting issue is it seems on linux the report ID byte is stripped
        by the driver, so we would not need to worry about it here. For windows, we must handle manually.

        :param buffer: list of bytes to send to device
        :param listen_for_reply: whether or not to listen for a reply
        :param timeout: timeout in seconds
        :return:
        """
        pass

    # sending and receiving commands, no operating system dependence
    def send_raw_command(self, buffer, listen_for_reply=False, timeout=5):
        """
        Send a raw command over USB, possibly including multiple packets.

        In contrast to send_command, this function does not generate the required header data. It deals with splitting
        one command into multiple packets and appropriately padding the supplied buffer.

        :param buffer: buffer to send. List of bytes.
        :param listen_for_reply: Boolean. Whether to wait for a reply form USB device
        :param timeout: time to wait for reply, in seconds
        :return: reply: a list of lists of bytes. Each list represents the response for a separate packet.
        """

        reply = []
        # handle sending multiple packets if necessary
        data_counter = 0
        while data_counter < len(buffer):

            # ensure data is correct length
            data_counter_next = data_counter + self.packet_length_bytes
            data_to_send = buffer[data_counter:data_counter_next]

            if len(data_to_send) < self.packet_length_bytes:
                # pad with zeros if necessary
                data_to_send += [0x00] * (self.packet_length_bytes - len(data_to_send))

            packet_reply = self._send_raw_packet(data_to_send, listen_for_reply, timeout)
            if packet_reply != []:
                reply.append(packet_reply)

            # increment for next packet
            data_counter = data_counter_next

        return reply

    def send_command(self, rw_mode, reply, command, data=(), sequence_byte=0x00):
        """
        Send USB command to DLP6500 DMD. Only works on Windows. For documentation of DMD commands, see dlpu018.pdf,
        available at http://www.ti.com/product/DLPC900/technicaldocuments

        DMD uses little endian byte order.

        They also use the convention that, when converting from binary to hex the MSB is the rightmost. i.e.
         \b11000000 = \x03. TODO: is this actually true??? Seems to not be true wrt to flag_byte, but true wrt to
         pattern defining bytes...maybe only care about this for data that is passed through the DMD?

        :param rw_mode: 'r' for read, or 'w' for write
        :param reply: boolean
        :param command: two byte integer
        :param data: data to be transmitted. List of integers, where each integer gives a byte
        :param sequence_byte: integer
        :return:
        """

        # construct header, 4 bytes long
        # first byte is flag byte
        flagstring = ''
        if rw_mode == 'r':
            flagstring += '1'
        elif rw_mode == 'w':
            flagstring += '0'
        else:
            raise ValueError("flagstring should be 'r' or 'w' but was '%s'" % flagstring)

        # second bit is reply
        if reply:
            flagstring += '1'
        else:
            flagstring += '0'

        # third bit is error bit
        flagstring += '0'
        # fourth and fifth reserved
        flagstring += '00'
        # 6-8 destination
        flagstring += '000'

        # first byte
        flag_byte = int(flagstring, 2)

        # second byte is sequence byte. This is used only to identify responses to given commands.

        # third and fourth are length of payload, respectively LSB and MSB bytes
        len_payload = len(data) + 2
        len_lsb, len_msb = struct.unpack('BB', struct.pack('H', len_payload))

        # get USB command bytes
        cmd_lsb, cmd_msb = struct.unpack('BB', struct.pack('H', command))

        # this does not exactly correspond with what TI calls the header. It is a combination of
        # the report id_byte, the header, and the USB command bytes
        header = [flag_byte, sequence_byte, len_lsb, len_msb, cmd_lsb, cmd_msb]
        buffer = header + list(data)

        # print commands during debugging
        if self.debug:
            # get command name if possible
            # header
            print('header: ' + bin(header[0]), end=' ')
            for ii in range(1, len(header)):
                print("0x%0.2X" % header[ii], end=' ')
            print('')

            # get command name, if possible
            for k, v in self.command_dict.items():
                if v == command:
                    print(k + " (" + hex(command) + ") :", end=' ')
                    break

            # print contents of command
            for ii in range(len(data)):
                print("0x%0.2X" % data[ii], end=' ')
            print('')

        return self.send_raw_command(buffer, reply)

    def decode_command(self, buffer, mode='first-packet'):
        """
        Decode DMD command into constituent pieces
        """

        if mode == 'first-packet':
            flag_byte = bin(buffer[1])
            sequence_byte = hex(buffer[2])

            len_bytes = struct.pack('B', buffer[4]) + struct.pack('B', buffer[3])
            data_len = struct.unpack('H', len_bytes)[0]

            cmd = struct.pack('B', buffer[6]) + struct.pack('B', buffer[5])
            data = buffer[7:]
        elif mode == 'nth-packet':
            flag_byte = None
            sequence_byte = None
            len_bytes = None
            data_len = None
            cmd = None
            data = buffer[1:]
        else:
            raise ValueError("mode must be 'first-packet' or 'nth-packet', but was '%s'" % mode)

        return flag_byte, sequence_byte, data_len, cmd, data

    def decode_flag_byte(self, flag_byte):
        """
        Get parameters from flags set in the flag byte
        :param flag_byte:
        :return:
        """

        errs = [2 ** ii & flag_byte != 0 for ii in range(5, 8)]
        err_names = ['error', 'host requests reply', 'read transaction']
        result = {}
        for e, en in zip(errs, err_names):
            result[en] = e

        return result

    def decode_response(self, buffer):
        """
        Parse USB response from DMD into useful info

        :param buffer:
        :return:
        """

        flag_byte = buffer[0]
        response = self.decode_flag_byte(flag_byte)

        sequence_byte = buffer[1]

        # len of data
        len_bytes = struct.pack('B', buffer[2]) + struct.pack('B', buffer[3])
        data_len = struct.unpack('<H', len_bytes)[0]

        # data
        data = buffer[4:4 + data_len]

        # all information
        response.update({'sequence byte': sequence_byte, 'data': data})

        return response

    # check DMD info
    def read_error_code(self):
        """
        # todo: DMD complains about this command...says invalid command number 0x100
        Retrieve error code number from last executed command
        """

        buffer = self.send_command('w', True, 0x0100)[0]
        resp = self.decode_response(buffer)
        err_code = resp['data'][0]

        error_type = 'not defined'
        for k, v in self.err_dictionary.items():
            if v == err_code:
                error_type = k
                break

        return error_type, err_code

    def read_error_description(self):
        """
        Retrieve error code description for last error
        :return:
        """
        buffer = self.send_command('r', True, 0x0101)[0]
        resp = self.decode_response(buffer)

        # read until find C style string termination byte, \x00
        # NOTE: when new error messages are written to the DMD buffer, they are written over previous messages.
        # If the new error messages is shorter than the previous one, the remaining unoverwritten part remains
        # in the buffer. Do not be alarmed!
        err_description = ''
        for ii, d in enumerate(resp['data']):
            if d == 0:
                break

            err_description += chr(d)

        return err_description

    def get_hw_status(self):
        """
        Get hardware status of DMD
        :return:
        """
        buffer = self.send_command('r', True, 0x1A0A)[0]
        resp = self.decode_response(buffer)

        errs = [(2**ii & resp['data'][0]) != 0 for ii in range(8)]
        err_names = ['internal initialization success', 'incompatible controller or DMD',
                     'DMD rest controller error', 'forced swap error', 'slave controller present',
                     'reserved', 'sequence abort status error', 'sequencer error']
        result = {}
        for e, en in zip(errs, err_names):
            result[en] = e

        return result

    def get_system_status(self):
        """
        Get status of internal memory test
        :return:
        """
        buffer = self.send_command('r', True, 0x1A0B)[0]
        resp = self.decode_response(buffer)

        return {'internal memory test passed': bool(resp['data'][0])}

    def get_main_status(self):
        """
        Get DMD main status
        :return:
        """
        # todo: which byte gives info? first data byte?
        buffer = self.send_command('r', True, 0x1A0C)[0]
        resp = self.decode_response(buffer)

        errs = [2 ** ii & resp['data'][0] != 0 for ii in range(8)]
        err_names = ['DMD micromirrors are parked', 'sequencer is running normally', 'video is frozen',
                     'external video source is locked', 'port 1 syncs valid', 'port 2 syncs valid',
                     'reserved', 'reserved']
        result = {}
        for e, en in zip(errs, err_names):
            result[en] = e

        return result

    def get_firmware_version(self):
        """
        Get firmware version information from DMD
        :return:
        """
        buffer = self.send_command('r', True, 0x0205)[0]
        resp = self.decode_response(buffer)

        app_version = resp['data'][0:4]
        app_patch = struct.unpack('<H', b"".join([b.to_bytes(1, 'big') for b in app_version[0:2]]))[0]
        app_minor = app_version[2]
        app_major = app_version[3]
        app_version_str = '%d.%d.%d' % (app_major, app_minor, app_patch)

        api_version = resp['data'][4:8]
        api_patch = struct.unpack('<H', b"".join([b.to_bytes(1, 'big') for b in api_version[0:2]]))[0]
        api_minor = api_version[2]
        api_major = api_version[3]
        api_version_str = '%d.%d.%d' % (api_major, api_minor, api_patch)

        software_config_revision = resp['data'][8:12]
        swc_patch = struct.unpack('<H', b"".join([b.to_bytes(1, 'big') for b in software_config_revision[0:2]]))[0]
        swc_minor = software_config_revision[2]
        swc_major = software_config_revision[3]
        swc_version_str = '%d.%d.%d' % (swc_major, swc_minor, swc_patch)

        sequencer_config_revision = resp['data'][12:16]
        sqc_patch = struct.unpack('<H', b"".join([b.to_bytes(1, 'big') for b in sequencer_config_revision[0:2]]))[0]
        sqc_minor = sequencer_config_revision[2]
        sqc_major = sequencer_config_revision[3]
        sqc_version_str = '%d.%d.%d' % (sqc_major, sqc_minor, sqc_patch)

        result = {'app version': app_version_str, 'api version': api_version_str,
                  'software configuration revision': swc_version_str,
                  'sequence configuration revision': sqc_version_str}

        return result

    def get_firmware_type(self):
        """
        Get DMD type and firmware tag
        :return:
        """
        buffer = self.send_command('r', True, 0x0206)[0]
        resp = self.decode_response(buffer)

        dmd_type_flag = resp['data'][0]
        if dmd_type_flag == 0:
            dmd_type = 'unknown'
        elif dmd_type_flag == 1:
            dmd_type = 'DLP6500'
        elif dmd_type_flag == 2:
            dmd_type = 'DLP9000'
        else:
            raise ValueError("Unknown DMD type index %d. Expected 1 or 2", dmd_type_flag)

        # in principle could receive two packets. TODO: handle that case
        firmware_tag = ''
        for d in resp['data'][1:]:
            # terminate on \x00. This represents end of string (e.g. in C language)
            if d == 0:
                break

            firmware_tag += chr(d)

        return {'dmd type': dmd_type, 'firmware tag': firmware_tag}

    # trigger setup
    def set_trigger_out(self, trigger_number=1, invert=False, rising_edge_delay_us=0, falling_edge_delay_us=0):
        """
        Set DMD output trigger delays and polarity. Trigger 1 is the "advance frame" trigger and trigger 2 is the
        "enable" trigger
        # todo: test this function
        :param trigger_number:
        :param invert:
        :param rising_edge_delay_us:
        :param falling_edge_delay_us:
        :return:
        """

        if rising_edge_delay_us < -20 or rising_edge_delay_us > 20e3:
            raise ValueError('rising edge delay must be in range -20 -- 20000us')

        if falling_edge_delay_us < -20 or falling_edge_delay_us > 20e3:
            raise ValueError('falling edge delay must be in range -20 -- 20000us')

        if invert:
            assert rising_edge_delay_us >= falling_edge_delay_us

        # data
        trig_byte = [int(invert)]
        rising_edge_bytes = struct.unpack('BB', struct.pack('<h', rising_edge_delay_us))
        falling_edge_bytes = struct.unpack('BB', struct.pack('<h', falling_edge_delay_us))
        data = trig_byte + rising_edge_bytes + falling_edge_bytes

        if trigger_number == 1:
            resp = self.send_command('w', True, 0x1A1D, data)[0]
        elif trigger_number == 2:
            resp = self.send_command('w', True, 0x1A1E, data)[0]
        else:
            raise ValueError('trigger_number must be 1 or 2')

        return resp

    def get_trigger_in1(self):
        """
        Query information about trigger 1 ("advance frame" trigger)
        :return delay_us:
        :return mode:
        """
        buffer = self.send_command('r', True, 0x1A35, [])[0]
        resp = self.decode_response(buffer)
        data = resp['data']
        delay_us, = struct.unpack('<H', struct.pack('B', data[0]) + struct.pack('B', data[1]))
        mode = data[2]

        return delay_us, mode

    def set_trigger_in1(self, delay_us=105, edge_to_advance='rising'):
        """
        Set delay and pattern advance edge for trigger input 1 ("advance frame" trigger)

        Trigger input 1 is used to advance the pattern displayed on the DMD, provided trigger_in2 is
        in the appropriate state

        :param int delay_us:
        :param str edge_to_advance: 'rsiing' or 'falling'
        :return:
        """

        if delay_us < 104:
            raise ValueError('delay time must be 105us or longer.')

        # todo: is this supposed to be a signed or unsigned integer.
        delay_byte = list(struct.unpack('BB', struct.pack('<H', delay_us)))

        if edge_to_advance == 'rising':
            advance_byte = [0x00]
        elif edge_to_advance == 'falling':
            advance_byte = [0x01]
        else:
            raise ValueError("edge_to_advance must be 'rising' or 'falling', but was '%s'" % edge_to_advance)

        return self.send_command('w', True, 0x1A35, delay_byte + advance_byte)

    def get_trigger_in2(self):
        """
        Query polarity of trigger in 2 ("enable" trigger)
        :return:
        """
        buffer = self.send_command('r', True, 0x1A36, [])[0]
        resp = self.decode_response(buffer)
        mode = resp['data'][0]
        return mode

    def set_trigger_in2(self, edge_to_start='rising'):
        """
        Set polarity to start/stop pattern on for input trigger 2 ("enable" trigger)

        Trigger input 2 is used to start or stop the DMD pattern display.
        :param edge_to_start:
        :return:
        """
        if edge_to_start == 'rising':
            start_byte = [0x00]
        elif edge_to_start == 'falling':
            start_byte = [0x01]
        else:
            raise ValueError("edge_to_start must be 'rising' or 'falling', but was '%s'" % edge_to_start)

        return self.send_command('w', False, 0x1A36, start_byte)

    # sequence start stop
    def set_pattern_mode(self, mode='on-the-fly'):
        """
        Change the DMD display mode

        "DISP_MODE" according to GUI

        :param mode: string. 'video', 'pre-stored', 'video-pattern', or 'on-the-fly'
        """
        if mode == 'video':
            data = [0x00]
        elif mode == 'pre-stored':
            data = [0x01]
        elif mode == 'video-pattern':
            data = [0x02]
        elif mode == 'on-the-fly':
            data = [0x03]
        else:
            raise ValueError("mode '%s' is not allowed" % mode)

        return self.send_command('w', True, 0x1A1B, data)

    def start_stop_sequence(self, cmd):
        """
        Start, stop, or pause a pattern sequence

        "PAT_START_STOP" according to GUI

        :param cmd: string. 'start', 'stop' or 'pause'
        """
        if cmd == 'start':
            data = [0x02]
            seq_byte = 0x08
        elif cmd == 'stop':
            data = [0x00]
            seq_byte = 0x05
        elif cmd == 'pause':
            data = [0x01]
            seq_byte = 0x00 # todo: check this from packet sniffer
        else:
            raise ValueError("cmd must be 'start', 'stop', or 'pause', but was '%s'" % cmd)

        return self.send_command('w', False, 0x1A24, data, sequence_byte=seq_byte)

    #######################################
    # commands for working with patterns
    #######################################
    def pattern_display_lut_configuration(self, num_patterns, num_repeat=0):
        """
        Controls the execution of patterns stored in the lookup table (LUT). Before executing this command,
        stop the current pattern sequence.

        "PAT_CONFIG" according to GUI

        :param num_patterns: Number of LUT entries, 0-511
        :param num_repeat: number of times to repeat the pattern sequence
        :return:
        """

        # assert num_patterns < 256
        num_patterns_bytes = list(struct.unpack('BB', struct.pack('<H', num_patterns)))
        num_repeats_bytes = list(struct.unpack('BBBB', struct.pack('<I', num_repeat)))

        # return self.send_command('w', False, 0x1A31, data=num_patterns_bytes + num_repeats_bytes)
        return self.send_command('w', True, 0x1A31, data=num_patterns_bytes + num_repeats_bytes)

    def pattern_display_lut_definition(self, sequence_position_index, exposure_time_us=105, dark_time_us=0,
                                       wait_for_trigger=True, clear_pattern_after_trigger=False, bit_depth=1,
                                       disable_trig_2=True, stored_image_index=0, stored_image_bit_index=0):
        """
        Define parameters for pattern used in on-the-fly mode. This command is listed as "MBOX_DATA"
         in the DLPLightcrafter software GUI.

        Display mode and pattern display LUT configuration must be set before sending pattern LUT definition data.
        These can be set using set_pattern_mode() and pattern_display_lut_configuration() respectively.  If the pattern
        display data input source is set to streaming the image indices do not need to be set.

        When uploading 1 bit image, each set of 24 images are first combined to a single 24 bit RGB image. pattern_index
        refers to which 24 bit RGB image a pattern is in, and pattern_bit_index refers to which bit of that image (i.e.
        in the RGB bytes, it is stored in.

        :param sequence_position_index:
        :param exposure_time_us:
        :param dark_time_us:
        :param wait_for_trigger:
        :param clear_pattern_after_trigger:
        :param bit_depth: 1, 2, 4, 8
        :param disable_trig_2: (disable "enable" trigger)
        :param stored_image_index:
        :param stored_image_bit_index: index of the RGB image (in DMD memory) storing the given pattern
        this index tells which bit to look at in that image. This should be 0-23
        """

        # assert pattern_index < 256 and pattern_index >= 0

        pattern_index_bytes = list(struct.unpack('BB', struct.pack('<H', sequence_position_index)))
        # actually can only use the first 3 bytes
        exposure_bytes = list(struct.unpack('BBBB', struct.pack('<I', exposure_time_us)))[:-1]

        # next byte contains various different information
        # first bit gives
        misc_byte_str = ''

        if clear_pattern_after_trigger:
            misc_byte_str += '1'
        else:
            misc_byte_str += '0'

        # next three bits give bit depth, integer 1 = 000, ..., 8 = 111
        if bit_depth != 1:
            raise NotImplementedError('bit_depths other than 1 not implemented.')
        misc_byte_str += '000'

        # next 3 give LED's enabled or disabled. Always disabled
        # todo: think usually GUI sends command to 100 for this?
        misc_byte_str += '100'

        if wait_for_trigger:
            misc_byte_str += '1'
        else:
            misc_byte_str += '0'

        misc_byte = [int(misc_byte_str[::-1], 2)]

        dark_time_bytes = list(struct.unpack('BB', struct.pack('<H', dark_time_us))) + [0]
        if disable_trig_2:
            trig2_output_bytes = [0x00]
        else:
            trig2_output_bytes = [0x01]

        # actually bits 0:10
        img_pattern_index_byte = [stored_image_index]
        # todo: how to set this byte?
        # actually only bits 11:15
        # don't really understand why, but this is what GUI sets for these...
        # NOTE: can reuse a pattern in the LUT by setting this bit to the same as another
        # in that case would not need to send the PATMEM_LOAD_INIT_MASTER or -TAMEM_LOAD_DATA_MASTER commands
        pattern_bit_index_byte = [8 * stored_image_bit_index]

        data = pattern_index_bytes + exposure_bytes + misc_byte + \
               dark_time_bytes + trig2_output_bytes + img_pattern_index_byte + pattern_bit_index_byte

        return self.send_command('w', True, 0x1A34, data)

    def init_pattern_bmp_load(self, pattern_length, pattern_index):
        """
        Initialize pattern BMP load command.

        DMD GUI calls this "PATMEM_LOAD_INIT_MASTER"

        When the initialize pattern BMP load command is issued, the patterns in the flash are not used until the pattern
        mode is disabled by command. This command should be followed by the pattern_bmp_load() command to load images.
        The images should be loaded in reverse order.
        """

        # packing and unpacking bytes doesn't do anything...but for consistency...
        index_bin = list(struct.unpack('BB', struct.pack('<H', pattern_index)))
        num_bytes = list(struct.unpack('BBBB', struct.pack('<I', pattern_length)))
        data = index_bin + num_bytes

        # return self.send_command('w', False, 0x1A2A, data=data)
        return self.send_command('w', True, 0x1A2A, data=data)

    def pattern_bmp_load(self, compressed_pattern, compression_mode):
        """
        Load DMD pattern data for use in pattern on-the-fly mode. To load all necessary data to DMD correctly,
        invoke this from upload_pattern_sequence()

        The DMD GUI software calls this command "PATMEM_LOAD_DATA_MASTER"

        Some complications to this command: the DMD can only deal with 512 bytes at a time. So including the packet
        header, can only send 512 - len(header) - len_command_data_bytes.
        since the header is 6 bytes and the length of the data is represented using 2 bytes, there are 504 data bytes
        After this, have to send a new command.

        """
        max_cmd_payload = 504

        # get the header
        # Note: taken directly from sniffer of the TI GUI
        signature_bytes = [0x53, 0x70, 0x6C, 0x64]
        width_byte = list(struct.unpack('BB', struct.pack('<H', 1920)))
        height_byte = list(struct.unpack('BB', struct.pack('<H', 1080)))
        # Number of bytes in encoded image_data
        num_encoded_bytes = list(struct.unpack('BBBB', struct.pack('<I', len(compressed_pattern))))
        reserved_bytes = [0xFF] * 8  # reserved
        bg_color_bytes = [0x00] * 4  # BG color BB, GG, RR, 00

        # encoding 0 none, 1 rle, 2 erle
        if compression_mode == 'none':
            encoding_byte = [0x00]
        elif compression_mode == 'rle':
            encoding_byte = [0x01]
        elif compression_mode == 'erle':
            encoding_byte = [0x02]
        else:
            raise ValueError("compression_mode must be 'none', 'rle', or 'erle' but was '%s'" % compression_mode)

        general_data = signature_bytes + width_byte + height_byte + num_encoded_bytes + \
                 reserved_bytes + bg_color_bytes + [0x01] + encoding_byte + \
                 [0x01] + [0x00] * 2 + [0x01] + [0x00] * 18 # reserved

        data = general_data + compressed_pattern

        # send multiple commands, each of maximum size 512 bytes including header
        data_index = 0
        command_index = 0
        while data_index < len(data):
            # slice data to get block to send in this command
            data_index_next = np.min([data_index + max_cmd_payload, len(data)])
            data_current = data[data_index:data_index_next]

            # len of current data block
            data_len_bytes = list(struct.unpack('BB', struct.pack('<H', len(data_current))))

            # send command
            self.send_command('w', False, 0x1A2B, data=data_len_bytes + data_current)

            data_index = data_index_next
            command_index += 1

    def upload_pattern_sequence(self, patterns, exp_times, dark_times, triggered=False,
                                clear_pattern_after_trigger=True, bit_depth=1, num_repeats=0, compression_mode='erle',
                                combine_images=True):
        """
        Upload on-the-fly pattern sequence to DMD.
        # todo: seems I need to call set_pattern_sequence() after this command to actually get sequence running. Why?

        WARNING: When using triggered operations, the DLP6500 behaves differently depending on the state of the trigger
        in signals when this command is issued. If the trigger in signals are HIGH, then the patterns will be displayed
        when the trigger is high. If the trigger in signals are LOW, then the patterns will be displayed when
        the trigger is low.

        This command is based on Table 5-3 in the DLP programming manual

        :param patterns: N x Ny x Nx NumPy array of uint8
        :param list[int] exp_times: exposure times in us. Either a single uint8 number, or a list the same
         length as the number of patterns. >=105us
        :param list[int] dark_times: dark times in us. Either a single uint8 number or a list the same length
         as the number of patterns
        :param bool triggered: Whether or not DMD should wait to be triggered to display the next pattern
        :param bool clear_pattern_after_trigger: Whether or not to keep displaying the pattern at the
         end of exposure time,
        i.e. during time while DMD is waiting for the next trigger.
        :param int bit_depth: Bit depth of patterns
        :param int num_repeats: Number of repeats. 0 means infinite.
        :param str compression_mode: 'erle', 'rle', or 'none'
        :param bool combine_images:
        :return stored_image_indices, stored_bit_indices: image and bit indices where each image was stored
        """
        # #########################
        # check arguments
        # #########################
        if patterns.dtype != np.uint8:
            raise ValueError('patterns must be of dtype uint8')

        if patterns.ndim == 2:
            patterns = np.expand_dims(patterns, axis=0)

        npatterns = len(patterns)

        # if only one exp_times, apply to all patterns
        if not isinstance(exp_times, (list, np.ndarray)):
            exp_times = [exp_times]

        if not all(list(map(lambda t: isinstance(t, int), exp_times))):
            raise ValueError("exp_times must be a list of integers")

        if patterns.shape[0] > 1 and len(exp_times) == 1:
            exp_times = exp_times * patterns.shape[0]

        # if only one dark_times, apply to all patterns
        if isinstance(dark_times, int):
            dark_times = [dark_times]

        if not all(list(map(lambda t: isinstance(t, int), dark_times))):
            raise ValueError("dark_times must be a list of integers")

        if patterns.shape[0] > 1 and len(dark_times) == 1:
            dark_times = dark_times * patterns.shape[0]

        # #########################
        # #########################

        # need to issue stop before changing mode. Otherwise DMD will sometimes lock up and not be responsive.
        self.start_stop_sequence('stop')
        # buffer = self.start_stop_sequence('stop')[0]
        # resp = self.decode_response(buffer)
        # if resp['error']:
        #     print(self.read_error_description())

        # set to on-the-fly mode
        buffer = self.set_pattern_mode('on-the-fly')[0]
        resp = self.decode_response(buffer)
        if resp['error']:
            print(self.read_error_description())

        # stop any currently running sequences
        # note: want to stop after changing pattern mode, because otherwise may throw error
        self.start_stop_sequence('stop')
        # buffer = self.start_stop_sequence('stop')[0]
        # resp = self.decode_response(buffer)
        # if resp['error']:
        #     print(self.read_error_description())

        # set image parameters for look up table
        # When uploading 1 bit image, each set of 24 images are first combined to a single 24 bit RGB image. pattern_index
        # refers to which 24 bit RGB image a pattern is in, and pattern_bit_index refers to which bit of that image (i.e.
        # in the RGB bytes, it is stored in.
        # todo: decide if is smaller to send compressed as 24 bit RGB or individual image...
        stored_image_indices = np.zeros(npatterns, dtype=int)
        stored_bit_indices = np.zeros(npatterns, dtype=int)
        for ii, (p, et, dt) in enumerate(zip(patterns, exp_times, dark_times)):
            stored_image_indices[ii] = ii // 24
            stored_bit_indices[ii] = ii % 24
            buffer = self.pattern_display_lut_definition(ii, exposure_time_us=et, dark_time_us=dt,
                                                         wait_for_trigger=triggered,
                                                         clear_pattern_after_trigger=clear_pattern_after_trigger,
                                                         bit_depth=bit_depth,
                                                         stored_image_index=stored_image_indices[ii],
                                                         stored_image_bit_index=stored_bit_indices[ii])[0]
            resp = self.decode_response(buffer)
            if resp['error']:
                print(self.read_error_description())

        buffer = self.pattern_display_lut_configuration(npatterns, num_repeats)[0]
        resp = self.decode_response(buffer)
        if resp['error']:
            print(self.read_error_description())

        # can combine images if bit depth = 1
        # todo: test if things work if I don't combine images in the 24bit format
        if combine_images:
            if bit_depth == 1:
                patterns = combine_patterns(patterns)
            else:
                raise NotImplementedError("Combining multiple images into a 24-bit RGB image is only"
                                          " implemented for bit depth 1.")

        # compress and load images
        # images must be loaded in backwards order according to programming manual
        for ii, dmd_pattern in reversed(list(enumerate(patterns))):
            print("sending pattern %d/%d" % (ii + 1, len(patterns)))

            if compression_mode == 'none':
                raise NotImplementedError("compression mode 'none' has not been tested.")
                compressed_pattern = np.packbits(dmd_pattern.ravel())
            elif compression_mode == 'rle':
                raise NotImplementedError("compression mode 'rle' as not been tested.")
                compressed_pattern = encode_rle(dmd_pattern)
            elif compression_mode == 'erle':
                compressed_pattern = encode_erle(dmd_pattern)

            # pattern has an additional 48 bytes which must be sent also.
            # todo: Maybe I should nest the call to init_pattern_bmp_load in pattern_bmp_load?
            buffer = self.init_pattern_bmp_load(len(compressed_pattern) + 48, pattern_index=ii)[0]
            resp = self.decode_response(buffer)
            if resp['error']:
                print(self.read_error_description())

            self.pattern_bmp_load(compressed_pattern, compression_mode)

        # this PAT_CONFIG command is necessary, otherwise subsequent calls to set_pattern_sequence() will not behave
        # as expected.
        buffer = self.pattern_display_lut_configuration(npatterns, num_repeats)[0]
        resp = self.decode_response(buffer)
        if resp['error']:
            print(self.read_error_description())

        # start sequence
        self.start_stop_sequence('start')
        # buffer = self.start_stop_sequence('start')[0]
        # resp = self.decode_response(buffer)
        # if resp['error']:
        #     print(self.read_error_description())

        # test

        # some weird behavior where wants to be STOPPED before starting triggered sequence. This seems to happen
        # intermittently. Probably due to some other DMD setting that I'm not aware of?
        # todo: is this still true? One problem is the way the patterns respond depends on what the DMD's
        # trigger state is when loading
        if triggered:
            self.start_stop_sequence('stop')

        return stored_image_indices, stored_bit_indices

    def set_pattern_sequence(self, image_indices, bit_indices, exp_times, dark_times, triggered=False,
                             clear_pattern_after_trigger=True, bit_depth=1, num_repeats=0, mode='pre-stored'):
        """
        Setup pattern sequence from patterns previously stored in DMD memory, either in on-the-fly pattern mode,
         or in pre-stored pattern mode

        :param list[int] image_indices:
        :param list[int] bit_indices:
        :param int exp_times:
        :param int dark_times:
        :param bool triggered:
        :param bool clear_pattern_after_trigger:
        :param int bit_depth:
        :param int num_repeats: number of repeats. 0 repeats means repeat continuously.
        :param str mode: pre-stored or
        :return:
        """
        # #########################
        # check arguments
        # #########################
        if isinstance(image_indices, int) or np.issubdtype(type(image_indices), np.integer):
            image_indices = [image_indices]
        elif isinstance(image_indices, np.ndarray):
            image_indices = list(image_indices)

        if isinstance(bit_indices, int) or np.issubdtype(type(bit_indices), np.integer):
            bit_indices = [bit_indices]
        elif isinstance(bit_indices, np.ndarray):
            bit_indices = list(bit_indices)

        if len(image_indices) != len(bit_indices):
            raise ValueError("image_indices and bit_indices must be the same length.")

        nimgs = len(image_indices)

        if mode == 'on-the-fly' and 0 not in bit_indices:
            raise ValueError("Known issue (not with this code, but with DMD) that if 0 is not included in the bit"
                             "indices, then the patterns displayed will not correspond with the indices supplied.")

        # if only one exp_times, apply to all patterns
        if isinstance(exp_times, int):
            exp_times = [exp_times]

        if not all(list(map(lambda t: isinstance(t, int), exp_times))):
            raise ValueError("exp_times must be a list of integers")

        if nimgs > 1 and len(exp_times) == 1:
            exp_times = exp_times * nimgs

        # if only one dark_times, apply to all patterns
        if isinstance(dark_times, int):
            dark_times = [dark_times]

        if not all(list(map(lambda t: isinstance(t, int), dark_times))):
            raise ValueError("dark_times must be a list of integers")

        if nimgs > 1 and len(dark_times) == 1:
            dark_times = dark_times * nimgs

        # #########################
        # #########################
        # need to issue stop before changing mode. Otherwise DMD will sometimes lock up and not be responsive.
        self.start_stop_sequence('stop')

        # set to pattern mode
        buffer = self.set_pattern_mode(mode)[0]
        resp = self.decode_response(buffer)
        if resp['error']:
            print(self.read_error_description())

        # stop any currently running sequences
        # note: want to stop after changing pattern mode, because otherwise may throw error
        self.start_stop_sequence('stop')

        # set image parameters for look up table_
        for ii, (et, dt) in enumerate(zip(exp_times, dark_times)):
            buffer = self.pattern_display_lut_definition(ii, exposure_time_us=et, dark_time_us=dt,
                                                         wait_for_trigger=triggered,
                                                         clear_pattern_after_trigger=clear_pattern_after_trigger,
                                                         bit_depth=bit_depth, stored_image_index=image_indices[ii],
                                                         stored_image_bit_index=bit_indices[ii])[0]
            resp = self.decode_response(buffer)
            if resp['error']:
                print(self.read_error_description())

        # PAT_CONFIG command
        buffer = self.pattern_display_lut_configuration(nimgs, num_repeat=num_repeats)[0]
        resp = self.decode_response(buffer)
        if resp['error']:
            print(self.read_error_description())

        # start sequence
        self.start_stop_sequence('start')

        # some weird behavior where wants to be STOPPED before starting triggered sequence
        if triggered:
            self.start_stop_sequence('stop')

    # batch files in firmware
    def get_fwbatch_name(self, batch_index):
        """
        Return name of batch file stored on firmware at batch_index

        :param batch_index:
        :return:
        """
        buffer = self.send_command('r', True, 0x1A14, [batch_index])[0]
        resp = self.decode_response(buffer)

        batch_name = ''
        for ii, d in enumerate(resp['data']):

            # if d == 0 and ii > 0:
            if d == 0:
                break

            batch_name += chr(d)

        return batch_name

    def execute_fwbatch(self, batch_index):
        """
        Execute batch file stored on firmware at index batch_index

        :param batch_index:
        :return:
        """
        return self.send_command('w', True, 0x1A15, [batch_index])

    def set_fwbatch_delay(self, delay_ms):
        """
        Set delay between batch file commands
        :param delay_ms:
        :return:
        """
        # todo: test this works!
        data = struct.unpack('BBBB', struct.pack('<I', delay_ms))
        data = list(data[:3])
        return self.send_command('w', True, 0x1A16, data)


class dlp6500win(dlp6500):
    """
    Class for handling dlp6500 on windows os
    """

    def __init__(self, vendor_id=0x0451, product_id=0xc900, debug=True):
        super(dlp6500win, self).__init__(vendor_id=vendor_id, product_id=product_id, debug=debug)

    def __del__(self):
        self.dmd.close()

    def _get_device(self, vendor_id, product_id):
        """
        Return handle to DMD. This command can contain OS dependent implenetation

        :param vendor_id: usb vendor id
        :param product_id: usb product id
        :return:
        """

        filter = pyhid.HidDeviceFilter(vendor_id=vendor_id, product_id=product_id)
        devices = filter.get_devices()
        self.dmd = devices[0]
        self.dmd.open()

        # variable for holding response of dmd
        self._response = []
        # strip off first return byte
        response_handler = lambda data: self._response.append(data[1:])
        self.dmd.set_raw_data_handler(response_handler)

    def _send_raw_packet(self, buffer, listen_for_reply=False, timeout=5):
        """
        Send a single USB packet.

        :param buffer: list of bytes to send to device
        :param listen_for_reply: whether or not to listen for a reply
        :param timeout: timeout in seconds
        :return: reply: a list of bytes
        """

        # ensure packet is correct length
        assert len(buffer) == self.packet_length_bytes

        report_id_byte = [0x00]

        # clear reply buffer before sending
        self._response = []

        # send
        reports = self.dmd.find_output_reports()
        reports[0].send(report_id_byte + buffer)

        # only wait for a reply if necessary
        if listen_for_reply:
            tstart = time.time()
            while self._response == []:
                time.sleep(0.1)
                tnow = time.time()

                if timeout is not None:
                    if (tnow - tstart) > timeout:
                        print('read command timed out')
                        break

        if self._response != []:
            reply = copy.deepcopy(self._response[0])
        else:
            reply = []

        return reply


class dlp6500ix(dlp6500):
    """
    Class for handling dlp6500 on linux os
    """

    def __init__(self, vendor_id=0x0451, product_id=0xc900, debug=True):
        raise NotImplementedError("dlp6500ix has not been fully implemented. The functions _get_device() and"
                                  " _send_raw_packet() need to be implemented.")
        super(dlp6500ix, self).__init__(vendor_id=vendor_id, product_id=product_id, debug=debug)

    def __del__(self):
        pass

    def _get_device(self, vendor_id, product_id):
        pass

    def _send_raw_packet(self, buffer, listen_for_reply=False, timeout=5):
        pass


class dlp6500dummy(dlp6500):
    """Dummy class, useful for testing command generation when no DMD is connected"""
    def __init__(self, vendor_id=0, product_id=0, debug=True):
        super(dlp6500dummy, self).__init__(vendor_id=vendor_id, product_id=product_id, debug=debug)

    def _get_device(self, vendor_id, product_id):
        pass

    def _send_raw_packet(self, buffer, listen_for_reply=False, timeout=5):
        pass

    def read_error_description(self):
        return [['']]
