import socket
import struct
from time import sleep

class UR5_RG2:
    """
    Class to control the ONRobot RG2 gripper connected to UR5e through Tool I/O via Modbus over TCP/IP.
    
    Follow the 'Hardware setup' section from: https://github.com/tonydle/ur_onrobot#hardware-setup.
    """

    def __init__(self, ur_ip: str, port: int = 54321, slave_address: int = 65, socket_timeout: float = 0.5):
        """
        ur_ip: IP address of the UR5e controller
        port: Port number for Modbus TCP/IP (default: 54321)
        slave_address: Modbus slave address of the RG2 gripper (default: 65)
        socket_timeout: Timeout for socket operations in seconds (default: 0.5)
        """
        self.slave_address = slave_address
        self.socket = socket.create_connection((ur_ip, port), timeout=socket_timeout)

    def __del__(self):
        self.socket.close()

    """
    | Type            | Address / Code | Name                          | Meaning / Units                     | Notes                                                        |
    | --------------- | -------------: | ----------------------------- | ----------------------------------- | ------------------------------------------------------------ |
    | Register        |            `0` | Target force                  | `0.1 N` per unit                    | Example: `100` = `10 N`                                      |
    | Register        |            `1` | Target width                  | `0.1 mm` per unit                   | Example: `800` = `80.0 mm`                                   |
    | Register        |            `2` | Control                       | Command bitfield / mode             | Used to start/stop gripper motion                            |
    | Register        |          `258` | Fingertip offset              | Device-specific scaling             |                                                              |
    | Register        |          `267` | Actual width                  | `0.1 mm` per unit                   | Example: `900` = `90.0 mm`                                   |
    | Register        |          `268` | Status                        | Status bitfield                     | motion/state flags                                           |
    | Control code    |            `1` | Grip                          | Start grip motion                   | Standard grip command                                        |
    | Control code    |            `8` | Stop                          | Stop current motion                 | Useful before issuing a new target                           |
    | Control code    |           `16` | Grip with offset              | Start motion using fingertip offset | Used for more precise positioning                            |
    | Modbus function |   `0x03` / `3` | Read Holding Registers        | Read one or more registers          |                                                              |
    | Modbus function |   `0x06` / `6` | Write Single Register         | Write one register                  | Supported for writable registers                             |
    | Modbus function |  `0x10` / `16` | Write Multiple Registers      | Write several consecutive registers |                                                              |
    | Modbus function |  `0x17` / `23` | Read/Write Multiple Registers | Combined read/write transaction     |                                                              |
    """
    
    def _crc16(self, data: bytearray) -> int:
        """
        Compute the Modbus CRC16 Redundancy Check.
        https://github.com/LacobusVentura/MODBUS-CRC16
        """
        table = (0x0000, 0xA001)
        crc = 0xFFFF

        for byte in data:
            crc ^= byte
            for _ in range(8):
                xor = crc & 0x01
                crc >>= 1
                crc ^= table[xor]

        return crc

    def _write_registers(self, values: list[int]) -> bytes:
        """
        Send a Modbus RTU function code 16 command to write multiple registers.
        """
        frame = bytearray([
            self.slave_address,                  # Modbus slave address
            0x10,                   # write multiple registers (modbus function code 16)
            0x00, 0x00,             # start register = 0
            0x00, len(values),      # register count
            len(values) * 2,        # payload byte count
        ])

        for value in values:
            frame += struct.pack(">H", value) # big-endian 16-bit unsigned integer
        frame += struct.pack("<H", self._crc16(frame)) # little-endian CRC16 checksum

        self.socket.sendall(frame)
        return self.socket.recv(8)

    def _read_registers(self, start_address: int, count: int) -> list[int]:
        """
        Send a Modbus RTU function code 3 command to read holding registers.
        """
        frame = bytearray([
            self.slave_address,                          # Modbus slave address
            0x03,                           # Read holding registers (modbus function code 3)
            (start_address >> 8) & 0xFF,    # Start address (high byte)
            start_address & 0xFF,           # Start address (low byte)
            (count >> 8) & 0xFF,            # Register count (high byte)
            count & 0xFF,                   # Register count (low byte)
        ])
        frame += struct.pack("<H", self._crc16(frame)) # little-endian CRC16 checksum
        self.socket.sendall(frame)

        # FC03 response:
        # slave + function + byte_count + data + CRC
        expected_length = 5 + count * 2
        response = b""

        while len(response) < expected_length:
            chunk = self.socket.recv(expected_length - len(response))

            if not chunk:
                raise ConnectionError("Connection closed")

            response += chunk

        if response[0] != self.slave_address:
            raise RuntimeError("Wrong Modbus slave response")

        if response[1] & 0x80: # error response
            raise RuntimeError(
                f"Modbus exception code: {response[2]}"
            )

        registers: list[int] = []
        for i in range(count):
            offset = 3 + i * 2
            value = struct.unpack(">H", response[offset:offset + 2])[0]
            registers.append(value)

        return registers

    def is_busy(self) -> bool:
        """
        Check if the RG2 is currently moving.
        """
        status = self._read_registers(268, 1)[0] # register 268 = status
        return bool(status & 0x0001) # 0000000000000001 when busy, 0000000000000010 when idle
    
    def get_width(self) -> float:
        """
        Get the current measured gripper width in mm.
        """
        raw = self._read_registers(267, 1)[0] # Register 267 = measured opening
        return (raw / 10.0) - 10

    def move(self, width_mm: float|int, force_n: float|int = 10.0) -> None:
        """
        Move the RG2 to a target width in mm with a specified force in N.
        """
        width = int(round(width_mm * 10))
        force = int(round(force_n * 10))

        self._write_registers([force, width, 16])

    def stop(self) -> None:
        """
        Stop the RG2 motion immediately.
        """
        self._write_registers([0, 0, 8])

    def overwrite_move(self, width_mm: float|int, force_n: float|int = 10.0) -> None:
        """
        Interrupt current RG2 motion and immediately issue a new target.
        Does not cancel `move_and_wait`
        """
        self.stop()
        self.move(width_mm, force_n)

    def move_and_wait(self, width_mm: float|int, force_n: float|int = 10.0, sleep_interval: float = 0.05) -> None:
        """
        Move the RG2 to a target width in mm with a specified force in N and wait for completion.
        """
        self.move(width_mm, force_n)

        sleep(sleep_interval) 
        while self.is_busy():
            sleep(sleep_interval)

if __name__ == "__main__":

    ip_address = "192.168.1.103"
    port = 54321
    slave_address = 65

    rg2 = UR5_RG2(ip_address, port, slave_address)
    rg2.move(40.0, 10.0)
    rg2.move(80.0, 10.0)
    #rg2.overwrite_move(0, 10.0)
    rg2.move_and_wait(100, 10.0)
    print(f"Current width: {rg2.get_width()} mm")
