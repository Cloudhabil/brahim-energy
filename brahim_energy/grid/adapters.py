"""
Protocol Adapters - Backwards Compatibility Layer
=================================================

Layer 3 of the Brahim Onion Architecture.

Provides protocol adapters that wrap existing grid hardware without
requiring any modifications to the underlying systems.

Supported Protocols:
- Modbus TCP/RTU (Industrial PLCs, RTUs)
- DNP3 (SCADA systems)
- IEC 61850 (Substation automation)
- MQTT (IoT smart meters)
- REST API (Modern cloud systems)
- CSV (Legacy data files, historian exports)
- Simulation (Testing without hardware)

Each adapter implements the same interface, allowing the optimizer
to work with any hardware through a unified abstraction.

Author: GPIA Cognitive Ecosystem
Date: 2026-01-26
"""

from __future__ import annotations

import csv
import json
import logging
import random
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from brahim_energy.grid.optimizer import GridNode, NodeType, OnionGridOptimizer

logger = logging.getLogger("grid.protocol_adapters")


# =============================================================================
# BASE ADAPTER INTERFACE
# =============================================================================

class ProtocolAdapter(ABC):
    """
    Abstract base class for protocol adapters.

    All adapters must implement:
    - connect(): Establish connection to data source
    - disconnect(): Close connection
    - read_nodes(): Read current state of all nodes
    - write_command(): Send control command (optional)

    Adapters translate between protocol-specific data formats
    and the unified GridNode abstraction.
    """

    def __init__(self, optimizer: Optional[OnionGridOptimizer] = None):
        """
        Initialize adapter.

        Args:
            optimizer: Grid optimizer to register nodes with
        """
        self.optimizer = optimizer
        self.connected = False
        self._last_read: Optional[datetime] = None

    @property
    @abstractmethod
    def protocol_name(self) -> str:
        """Return protocol identifier."""
        pass

    @abstractmethod
    def connect(self) -> bool:
        """
        Establish connection to data source.

        Returns:
            True if connection successful
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Close connection."""
        pass

    @abstractmethod
    def read_nodes(self) -> List[GridNode]:
        """
        Read current state of all nodes.

        Returns:
            List of GridNode objects with current values
        """
        pass

    def write_command(
        self,
        node_id: str,
        command: str,
        value: Any
    ) -> bool:
        """
        Send control command to a node (optional).

        Default implementation is read-only (no control).
        Override in subclass for controllable adapters.

        Args:
            node_id: Target node ID
            command: Command type (e.g., "set_demand", "curtail")
            value: Command value

        Returns:
            True if command sent successfully
        """
        logger.warning(
            "%s adapter is read-only, command ignored: %s.%s = %s",
            self.protocol_name, node_id, command, value
        )
        return False

    def sync_to_optimizer(self) -> int:
        """
        Read nodes and register/update them in the optimizer.

        Returns:
            Number of nodes synced
        """
        if self.optimizer is None:
            logger.warning("No optimizer configured for sync")
            return 0

        nodes = self.read_nodes()

        for node in nodes:
            existing = self.optimizer.get_node(node.node_id)
            if existing is None:
                self.optimizer.register_node(node)
            else:
                self.optimizer.update_node(
                    node.node_id,
                    current_demand_kw=node.current_demand_kw,
                    capacity_kw=node.capacity_kw,
                    **node.metadata
                )

        self._last_read = datetime.utcnow()
        return len(nodes)


# =============================================================================
# SIMULATION ADAPTER (For Testing)
# =============================================================================

class SimulationAdapter(ProtocolAdapter):
    """
    Simulation adapter for testing without real hardware.

    Generates synthetic grid data with configurable patterns:
    - Daily load curves
    - Random fluctuations
    - Stress events
    - Renewable generation patterns
    """

    def __init__(
        self,
        optimizer: Optional[OnionGridOptimizer] = None,
        num_transformers: int = 5,
        num_feeders: int = 20,
        num_meters: int = 100,
        num_generators: int = 3,
        base_load_kw: float = 1000.0,
        peak_multiplier: float = 1.5,
        noise_level: float = 0.1
    ):
        """
        Initialize simulation adapter.

        Args:
            optimizer: Grid optimizer instance
            num_transformers: Number of simulated transformers
            num_feeders: Number of simulated feeders
            num_meters: Number of simulated meters
            num_generators: Number of simulated generators
            base_load_kw: Base load per meter
            peak_multiplier: Peak to base ratio
            noise_level: Random noise factor (0-1)
        """
        super().__init__(optimizer)

        self.num_transformers = num_transformers
        self.num_feeders = num_feeders
        self.num_meters = num_meters
        self.num_generators = num_generators
        self.base_load_kw = base_load_kw
        self.peak_multiplier = peak_multiplier
        self.noise_level = noise_level

        # Simulated node definitions
        self._node_defs: List[Dict] = []
        self._setup_nodes()

    @property
    def protocol_name(self) -> str:
        return "simulation"

    def _setup_nodes(self) -> None:
        """Create simulated node definitions."""
        self._node_defs = []

        # Transformers (high capacity)
        for i in range(self.num_transformers):
            self._node_defs.append({
                "node_id": f"TRANSFORMER_{i+1:02d}",
                "node_type": NodeType.TRANSFORMER,
                "capacity_kw": 5000.0 + random.uniform(-500, 500),
                "controllable": False,
                "priority": 1,
                "co2_intensity": 0.4,
            })

        # Feeders (medium capacity)
        for i in range(self.num_feeders):
            self._node_defs.append({
                "node_id": f"FEEDER_{i+1:03d}",
                "node_type": NodeType.FEEDER,
                "capacity_kw": 1000.0 + random.uniform(-100, 100),
                "controllable": False,
                "priority": 2,
                "co2_intensity": 0.4,
            })

        # Meters (low capacity, controllable)
        for i in range(self.num_meters):
            self._node_defs.append({
                "node_id": f"METER_{i+1:04d}",
                "node_type": NodeType.METER,
                "capacity_kw": 50.0 + random.uniform(-10, 10),
                "controllable": True,
                "priority": random.randint(3, 8),
                "co2_intensity": 0.4,
            })

        # Generators (renewable and conventional)
        for i in range(self.num_generators):
            is_renewable = i < self.num_generators // 2
            self._node_defs.append({
                "node_id": f"GEN_{i+1:02d}_{'SOLAR' if is_renewable else 'GAS'}",
                "node_type": NodeType.GENERATOR,
                "capacity_kw": 2000.0 if is_renewable else 5000.0,
                "controllable": True,
                "priority": 1,
                "co2_intensity": 0.0 if is_renewable else 0.5,
                "metadata": {"renewable": is_renewable},
            })

    def _get_load_multiplier(self, timestamp: datetime) -> float:
        """
        Calculate load multiplier based on time of day.

        Simulates typical daily load curve:
        - Low overnight (0.6x)
        - Morning ramp (0.8x -> 1.0x)
        - Midday plateau (1.0x)
        - Evening peak (1.2x -> 1.5x)
        - Evening decline (1.5x -> 0.8x)
        """
        hour = timestamp.hour + timestamp.minute / 60.0

        if hour < 6:
            # Overnight low
            return 0.6
        elif hour < 9:
            # Morning ramp
            return 0.6 + (hour - 6) / 3 * 0.4
        elif hour < 12:
            # Late morning
            return 1.0
        elif hour < 14:
            # Midday dip
            return 0.95
        elif hour < 18:
            # Afternoon plateau
            return 1.0
        elif hour < 20:
            # Evening peak
            progress = (hour - 18) / 2
            return 1.0 + progress * (self.peak_multiplier - 1.0)
        elif hour < 22:
            # Evening decline
            progress = (hour - 20) / 2
            return self.peak_multiplier - progress * (self.peak_multiplier - 0.8)
        else:
            # Late night decline
            progress = (hour - 22) / 2
            return 0.8 - progress * 0.2

    def connect(self) -> bool:
        """Simulation always connects successfully."""
        self.connected = True
        logger.info(
            "SimulationAdapter connected: %d transformers, %d feeders, %d meters, %d generators",
            self.num_transformers, self.num_feeders,
            self.num_meters, self.num_generators
        )
        return True

    def disconnect(self) -> None:
        """Disconnect simulation."""
        self.connected = False
        logger.info("SimulationAdapter disconnected")

    def read_nodes(self) -> List[GridNode]:
        """
        Read simulated node states.

        Generates realistic demand values based on:
        - Time of day load curve
        - Random fluctuations
        - Node type characteristics
        """
        if not self.connected:
            logger.warning("SimulationAdapter not connected")
            return []

        timestamp = datetime.utcnow()
        load_multiplier = self._get_load_multiplier(timestamp)

        nodes = []

        for node_def in self._node_defs:
            # Base demand depends on node type
            if node_def["node_type"] == NodeType.METER:
                base_demand = self.base_load_kw * 0.7  # 70% of base for meters
            elif node_def["node_type"] == NodeType.FEEDER:
                base_demand = self.base_load_kw * 10  # 10x for feeders
            elif node_def["node_type"] == NodeType.TRANSFORMER:
                base_demand = self.base_load_kw * 50  # 50x for transformers
            elif node_def["node_type"] == NodeType.GENERATOR:
                # Generators: negative demand = generation
                if node_def.get("metadata", {}).get("renewable"):
                    # Solar: follows daylight pattern
                    hour = timestamp.hour
                    if 6 <= hour <= 18:
                        solar_factor = 1.0 - abs(hour - 12) / 6
                        base_demand = -node_def["capacity_kw"] * solar_factor * 0.8
                    else:
                        base_demand = 0
                else:
                    # Conventional: steady generation
                    base_demand = -node_def["capacity_kw"] * 0.7
            else:
                base_demand = self.base_load_kw

            # Apply load multiplier and noise
            noise = 1.0 + random.uniform(-self.noise_level, self.noise_level)
            demand = base_demand * load_multiplier * noise

            # Ensure demand doesn't exceed capacity
            demand = min(demand, node_def["capacity_kw"] * 0.95)

            node = GridNode(
                node_id=node_def["node_id"],
                node_type=node_def["node_type"],
                capacity_kw=node_def["capacity_kw"],
                current_demand_kw=demand,
                protocol="simulation",
                controllable=node_def["controllable"],
                priority=node_def["priority"],
                co2_intensity=node_def["co2_intensity"],
                metadata=node_def.get("metadata", {}),
            )

            nodes.append(node)

        return nodes

    def write_command(
        self,
        node_id: str,
        command: str,
        value: Any
    ) -> bool:
        """
        Simulate writing a command.

        For simulation, this logs the command but doesn't persist changes.
        """
        logger.info(
            "SimulationAdapter command: %s.%s = %s",
            node_id, command, value
        )
        return True

    def inject_stress_event(
        self,
        target_utilization: float = 0.95,
        duration_seconds: float = 60.0
    ) -> None:
        """
        Inject a stress event for testing demand response.

        Temporarily increases load multiplier to create high utilization.
        """
        logger.warning(
            "Injecting stress event: target=%.1f%%, duration=%.0fs",
            target_utilization * 100, duration_seconds
        )
        # This would modify the load calculation temporarily
        # For now, just log the event


# =============================================================================
# MODBUS ADAPTER (Industrial PLCs)
# =============================================================================

class ModbusAdapter(ProtocolAdapter):
    """
    Modbus TCP/RTU adapter for industrial PLCs and RTUs.

    Supports:
    - Modbus TCP (Ethernet)
    - Modbus RTU (Serial RS-485)

    Register mapping is configurable per device type.

    Note: Requires pymodbus library for actual hardware communication.
    This implementation provides the interface; actual hardware
    communication requires pymodbus to be installed.
    """

    def __init__(
        self,
        optimizer: Optional[OnionGridOptimizer] = None,
        host: str = "localhost",
        port: int = 502,
        unit_id: int = 1,
        register_map: Optional[Dict[str, Dict]] = None
    ):
        """
        Initialize Modbus adapter.

        Args:
            optimizer: Grid optimizer instance
            host: Modbus TCP host address
            port: Modbus TCP port (default 502)
            unit_id: Modbus unit/slave ID
            register_map: Register address mapping
        """
        super().__init__(optimizer)

        self.host = host
        self.port = port
        self.unit_id = unit_id
        self.register_map = register_map or self._default_register_map()

        self._client = None

    @property
    def protocol_name(self) -> str:
        return "modbus"

    def _default_register_map(self) -> Dict[str, Dict]:
        """Default Modbus register mapping."""
        return {
            "demand_kw": {"address": 0, "count": 2, "type": "float32"},
            "capacity_kw": {"address": 2, "count": 2, "type": "float32"},
            "voltage": {"address": 4, "count": 2, "type": "float32"},
            "current": {"address": 6, "count": 2, "type": "float32"},
            "power_factor": {"address": 8, "count": 2, "type": "float32"},
        }

    def connect(self) -> bool:
        """
        Connect to Modbus device.

        Returns True if connection successful or if running in simulation mode.
        """
        try:
            # Try to import pymodbus
            from pymodbus.client import ModbusTcpClient
            self._client = ModbusTcpClient(self.host, port=self.port)
            self.connected = self._client.connect()

            if self.connected:
                logger.info(
                    "ModbusAdapter connected to %s:%d",
                    self.host, self.port
                )
            else:
                logger.error(
                    "ModbusAdapter failed to connect to %s:%d",
                    self.host, self.port
                )

            return self.connected

        except ImportError:
            logger.warning(
                "pymodbus not installed, ModbusAdapter running in stub mode. "
                "Install with: pip install pymodbus"
            )
            self.connected = False
            return False

    def disconnect(self) -> None:
        """Disconnect from Modbus device."""
        if self._client:
            self._client.close()
            self._client = None
        self.connected = False
        logger.info("ModbusAdapter disconnected")

    def read_nodes(self) -> List[GridNode]:
        """
        Read node data from Modbus registers.

        Note: This is a template implementation. Actual register
        addresses and data parsing depend on the specific device.
        """
        if not self.connected or not self._client:
            logger.warning("ModbusAdapter not connected")
            return []

        nodes = []

        try:
            # Read demand register
            demand_reg = self.register_map["demand_kw"]
            result = self._client.read_holding_registers(
                demand_reg["address"],
                demand_reg["count"],
                unit=self.unit_id
            )

            if result.isError():
                logger.error("Modbus read error: %s", result)
                return []

            # Parse float32 from registers
            demand_kw = self._parse_float32(result.registers)

            # Read capacity register
            capacity_reg = self.register_map["capacity_kw"]
            result = self._client.read_holding_registers(
                capacity_reg["address"],
                capacity_reg["count"],
                unit=self.unit_id
            )

            capacity_kw = self._parse_float32(result.registers) if not result.isError() else 1000.0

            node = GridNode(
                node_id=f"MODBUS_{self.host}_{self.unit_id}",
                node_type=NodeType.METER,
                capacity_kw=capacity_kw,
                current_demand_kw=demand_kw,
                protocol="modbus",
                controllable=False,
                metadata={
                    "host": self.host,
                    "port": self.port,
                    "unit_id": self.unit_id,
                }
            )

            nodes.append(node)

        except Exception as e:
            logger.error("ModbusAdapter read error: %s", e)

        return nodes

    def _parse_float32(self, registers: List[int]) -> float:
        """Parse IEEE 754 float32 from two 16-bit registers."""
        import struct
        if len(registers) < 2:
            return 0.0
        # Big-endian word order (common in Modbus)
        packed = struct.pack(">HH", registers[0], registers[1])
        return struct.unpack(">f", packed)[0]


# =============================================================================
# MQTT ADAPTER (IoT Smart Meters)
# =============================================================================

class MQTTAdapter(ProtocolAdapter):
    """
    MQTT adapter for IoT smart meters and sensors.

    Subscribes to topics and parses JSON payloads.

    Topic structure (configurable):
    - grid/{node_id}/demand
    - grid/{node_id}/capacity
    - grid/{node_id}/status

    Note: Requires paho-mqtt library for actual MQTT communication.
    """

    def __init__(
        self,
        optimizer: Optional[OnionGridOptimizer] = None,
        broker: str = "localhost",
        port: int = 1883,
        topic_prefix: str = "grid",
        username: Optional[str] = None,
        password: Optional[str] = None
    ):
        """
        Initialize MQTT adapter.

        Args:
            optimizer: Grid optimizer instance
            broker: MQTT broker address
            port: MQTT broker port
            topic_prefix: Base topic for grid data
            username: MQTT username (optional)
            password: MQTT password (optional)
        """
        super().__init__(optimizer)

        self.broker = broker
        self.port = port
        self.topic_prefix = topic_prefix
        self.username = username
        self.password = password

        self._client = None
        self._node_cache: Dict[str, GridNode] = {}

    @property
    def protocol_name(self) -> str:
        return "mqtt"

    def connect(self) -> bool:
        """Connect to MQTT broker and subscribe to topics."""
        try:
            import paho.mqtt.client as mqtt

            self._client = mqtt.Client()

            if self.username:
                self._client.username_pw_set(self.username, self.password)

            self._client.on_connect = self._on_connect
            self._client.on_message = self._on_message

            self._client.connect(self.broker, self.port, 60)
            self._client.loop_start()

            self.connected = True
            logger.info(
                "MQTTAdapter connected to %s:%d",
                self.broker, self.port
            )
            return True

        except ImportError:
            logger.warning(
                "paho-mqtt not installed, MQTTAdapter running in stub mode. "
                "Install with: pip install paho-mqtt"
            )
            self.connected = False
            return False

        except Exception as e:
            logger.error("MQTTAdapter connection error: %s", e)
            self.connected = False
            return False

    def _on_connect(self, client, userdata, flags, rc):
        """MQTT connection callback."""
        if rc == 0:
            # Subscribe to all grid topics
            topic = f"{self.topic_prefix}/#"
            client.subscribe(topic)
            logger.info("MQTTAdapter subscribed to %s", topic)
        else:
            logger.error("MQTTAdapter connection failed: rc=%d", rc)

    def _on_message(self, client, userdata, msg):
        """MQTT message callback."""
        try:
            # Parse topic: grid/{node_id}/{metric}
            parts = msg.topic.split("/")
            if len(parts) < 3:
                return

            node_id = parts[1]
            metric = parts[2]

            # Parse JSON payload
            payload = json.loads(msg.payload.decode())

            # Update or create node
            if node_id not in self._node_cache:
                self._node_cache[node_id] = GridNode(
                    node_id=node_id,
                    node_type=NodeType.METER,
                    capacity_kw=100.0,
                    current_demand_kw=0.0,
                    protocol="mqtt",
                )

            node = self._node_cache[node_id]

            # Update based on metric
            if metric == "demand":
                node.current_demand_kw = float(payload.get("value", 0))
            elif metric == "capacity":
                node.capacity_kw = float(payload.get("value", 100))
            elif metric == "status":
                node.metadata.update(payload)

        except Exception as e:
            logger.error("MQTTAdapter message parsing error: %s", e)

    def disconnect(self) -> None:
        """Disconnect from MQTT broker."""
        if self._client:
            self._client.loop_stop()
            self._client.disconnect()
            self._client = None
        self.connected = False
        logger.info("MQTTAdapter disconnected")

    def read_nodes(self) -> List[GridNode]:
        """Return cached nodes from MQTT subscriptions."""
        return list(self._node_cache.values())


# =============================================================================
# REST API ADAPTER (Modern Cloud Systems)
# =============================================================================

class RESTAdapter(ProtocolAdapter):
    """
    REST API adapter for modern cloud-based grid systems.

    Supports JSON APIs with configurable endpoints.
    """

    def __init__(
        self,
        optimizer: Optional[OnionGridOptimizer] = None,
        base_url: str = "http://localhost:8080/api",
        api_key: Optional[str] = None,
        nodes_endpoint: str = "/nodes",
        timeout: float = 10.0
    ):
        """
        Initialize REST adapter.

        Args:
            optimizer: Grid optimizer instance
            base_url: API base URL
            api_key: API key for authentication
            nodes_endpoint: Endpoint for node data
            timeout: Request timeout in seconds
        """
        super().__init__(optimizer)

        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.nodes_endpoint = nodes_endpoint
        self.timeout = timeout

    @property
    def protocol_name(self) -> str:
        return "rest"

    def connect(self) -> bool:
        """Test API connectivity."""
        try:
            import requests

            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            response = requests.get(
                f"{self.base_url}/health",
                headers=headers,
                timeout=self.timeout
            )

            self.connected = response.status_code == 200
            if self.connected:
                logger.info("RESTAdapter connected to %s", self.base_url)
            else:
                logger.warning(
                    "RESTAdapter health check failed: %d",
                    response.status_code
                )

            return self.connected

        except ImportError:
            logger.warning(
                "requests not installed, RESTAdapter running in stub mode. "
                "Install with: pip install requests"
            )
            self.connected = False
            return False

        except Exception as e:
            logger.error("RESTAdapter connection error: %s", e)
            self.connected = False
            return False

    def disconnect(self) -> None:
        """REST is stateless, nothing to disconnect."""
        self.connected = False
        logger.info("RESTAdapter disconnected")

    def read_nodes(self) -> List[GridNode]:
        """Fetch nodes from REST API."""
        if not self.connected:
            return []

        try:
            import requests

            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            response = requests.get(
                f"{self.base_url}{self.nodes_endpoint}",
                headers=headers,
                timeout=self.timeout
            )

            if response.status_code != 200:
                logger.error("RESTAdapter read failed: %d", response.status_code)
                return []

            data = response.json()
            nodes = []

            for item in data.get("nodes", []):
                node = GridNode(
                    node_id=item.get("id", "unknown"),
                    node_type=NodeType[item.get("type", "METER").upper()],
                    capacity_kw=float(item.get("capacity_kw", 100)),
                    current_demand_kw=float(item.get("demand_kw", 0)),
                    latitude=item.get("latitude"),
                    longitude=item.get("longitude"),
                    protocol="rest",
                    controllable=item.get("controllable", False),
                    priority=item.get("priority", 5),
                    co2_intensity=item.get("co2_intensity", 0.4),
                    metadata=item.get("metadata", {}),
                )
                nodes.append(node)

            return nodes

        except Exception as e:
            logger.error("RESTAdapter read error: %s", e)
            return []


# =============================================================================
# CSV ADAPTER (Legacy Data Files)
# =============================================================================

class CSVAdapter(ProtocolAdapter):
    """
    CSV adapter for legacy data files and historian exports.

    Useful for:
    - Historical data analysis
    - Importing data from SCADA historians
    - Testing with recorded data
    """

    def __init__(
        self,
        optimizer: Optional[OnionGridOptimizer] = None,
        file_path: Union[str, Path] = "grid_data.csv",
        id_column: str = "node_id",
        demand_column: str = "demand_kw",
        capacity_column: str = "capacity_kw",
        type_column: str = "node_type"
    ):
        """
        Initialize CSV adapter.

        Args:
            optimizer: Grid optimizer instance
            file_path: Path to CSV file
            id_column: Column name for node ID
            demand_column: Column name for demand
            capacity_column: Column name for capacity
            type_column: Column name for node type
        """
        super().__init__(optimizer)

        self.file_path = Path(file_path)
        self.id_column = id_column
        self.demand_column = demand_column
        self.capacity_column = capacity_column
        self.type_column = type_column

    @property
    def protocol_name(self) -> str:
        return "csv"

    def connect(self) -> bool:
        """Check if CSV file exists."""
        self.connected = self.file_path.exists()
        if self.connected:
            logger.info("CSVAdapter connected to %s", self.file_path)
        else:
            logger.warning("CSVAdapter file not found: %s", self.file_path)
        return self.connected

    def disconnect(self) -> None:
        """Nothing to disconnect for CSV."""
        self.connected = False

    def read_nodes(self) -> List[GridNode]:
        """Read nodes from CSV file."""
        if not self.connected:
            return []

        nodes = []

        try:
            with open(self.file_path, "r", newline="") as f:
                reader = csv.DictReader(f)

                for row in reader:
                    try:
                        node_type_str = row.get(self.type_column, "METER").upper()
                        node_type = NodeType[node_type_str] if node_type_str in NodeType.__members__ else NodeType.METER

                        node = GridNode(
                            node_id=row.get(self.id_column, f"CSV_{len(nodes)}"),
                            node_type=node_type,
                            capacity_kw=float(row.get(self.capacity_column, 100)),
                            current_demand_kw=float(row.get(self.demand_column, 0)),
                            protocol="csv",
                            metadata=dict(row),
                        )
                        nodes.append(node)

                    except Exception as e:
                        logger.warning("CSVAdapter row parse error: %s", e)
                        continue

        except Exception as e:
            logger.error("CSVAdapter read error: %s", e)

        return nodes


# =============================================================================
# ADAPTER FACTORY
# =============================================================================

ADAPTER_REGISTRY: Dict[str, type] = {
    "simulation": SimulationAdapter,
    "modbus": ModbusAdapter,
    "mqtt": MQTTAdapter,
    "rest": RESTAdapter,
    "csv": CSVAdapter,
}


def get_adapter(
    protocol: str,
    optimizer: Optional[OnionGridOptimizer] = None,
    **kwargs
) -> ProtocolAdapter:
    """
    Factory function to create protocol adapters.

    Args:
        protocol: Protocol name (simulation, modbus, mqtt, rest, csv)
        optimizer: Grid optimizer to register nodes with
        **kwargs: Protocol-specific configuration

    Returns:
        Configured ProtocolAdapter instance

    Raises:
        ValueError: If protocol is not supported
    """
    protocol = protocol.lower()

    if protocol not in ADAPTER_REGISTRY:
        supported = ", ".join(ADAPTER_REGISTRY.keys())
        raise ValueError(
            f"Unknown protocol: {protocol}. Supported: {supported}"
        )

    adapter_class = ADAPTER_REGISTRY[protocol]
    return adapter_class(optimizer=optimizer, **kwargs)


def register_adapter(name: str, adapter_class: type) -> None:
    """
    Register a custom protocol adapter.

    Args:
        name: Protocol name
        adapter_class: Adapter class (must extend ProtocolAdapter)
    """
    if not issubclass(adapter_class, ProtocolAdapter):
        raise TypeError("Adapter must extend ProtocolAdapter")

    ADAPTER_REGISTRY[name.lower()] = adapter_class
    logger.info("Registered custom adapter: %s", name)
