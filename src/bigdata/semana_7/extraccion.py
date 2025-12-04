import importlib
import requests
import datetime
from typing import Any, Dict, List, Optional

from pyspark.sql import SparkSession, DataFrame
import pyspark.sql.functions as F


class FootballApiClient:
    def __init__(
        self,
        spark: SparkSession,
        api_key:str,
        base_url: str = "https://free-api-live-football-data.p.rapidapi.com",
        host_header: str = "free-api-live-football-data.p.rapidapi.com"
    ):
        """
        Cliente para la Free API Live Football Data en RapidAPI.
        
        Parameters
        ----------
        spark : SparkSession
            Sesión de Spark de Databricks.
        api_key : str
            API key de RapidAPI (idealmente obtenida desde un secret).
        base_url : str
            URL base de la API.
        host_header : str
            Valor del header 'x-rapidapi-host'.
        """
        if len(api_key) > 0 :
            self.api_key = api_key
        else:
            raise ValueError("api_key no puede estar vacío")
        self.api_key = api_key
        self.spark = spark
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "x-rapidapi-host": host_header,
            "x-rapidapi-key": api_key
        }

    # ------------- Helpers internos -------------

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        url = f"{self.base_url}/{path.lstrip('/')}"
        print(f"Llamando a: {url}  con params={params}")

        resp = requests.get(url, headers=self.headers, params=params or {})

        print("Status code:", resp.status_code)
        print("Response preview:", resp.text[:500])

        resp.raise_for_status()
        return resp.json()

    # 🔁 NUEVO: helper simple que NO usa sparkContext
    def _records_to_df(self, records: List[Dict[str, Any]]) -> DataFrame:
        """
        Convierte una lista de diccionarios Python en un DataFrame de Spark
        sin usar sparkContext (compatible con serverless).
        """
        if not records:
            # Si viene vacío, devolvemos un DF vacío con inferSchema
            return self.spark.createDataFrame([], schema="ccode STRING, name STRING, localizedName STRING")
        return self.spark.createDataFrame(records)

    def _add_partition_cols(
        self,
        df: DataFrame,
        exec_date: Optional[datetime.date] = None
    ) -> DataFrame:
        if exec_date is None:
            exec_date = datetime.date.today()

        return (
            df.withColumn("year", F.lit(exec_date.year))
              .withColumn("month", F.lit(exec_date.month))
              .withColumn("day", F.lit(exec_date.day))
        )

    # ------------- COUNTRIES -------------
    def get_countries_df(
        self,
        with_partitions: bool = True,
        exec_date: Optional[datetime.date] = None
    ) -> DataFrame:
        """
        GET /football-get-all-countries
        JSON:
        {
          "status": "success",
          "response": {
            "countries": [ {...}, {...}, ... ]
          }
        }
        """
        data = self._get("football-get-all-countries")

        countries = (
            data.get("response", {})
                .get("countries", [])
        )

        df = self._records_to_df(countries)

        if with_partitions:
            df = self._add_partition_cols(df, exec_date)

        return df

    def get_teams_df(
        self,
        league_id: int,
        with_partitions: bool = True,
        exec_date: Optional[datetime.date] = None
    ) -> DataFrame:
        """
        GET /football-get-list-all-team?leagueid=<league_id>

        Respuesta suele ser algo como:
        {
          "status": "success",
          "response": {
            "teams": [ {...}, {...}, ... ]
          }
        }
        (si la clave no es 'teams', ajusta root_path tras ver el JSON)
        """
        params = {"leagueid": league_id}
        data = self._get("football-get-list-all-team", params=params)

        teams = self._extract_list(data, root_path=["response", "teams"])
        df = self._records_to_df(teams)

        # Guardamos la liga como columna
        df = df.withColumn("league_id", F.lit(league_id))

        if with_partitions:
            df = self._add_partition_cols(df, exec_date)
        return df

    def get_team_league_df(
        self,
        team_id: int,
        with_partitions: bool = True,
        exec_date: Optional[datetime.date] = None
    ) -> DataFrame:
        """
        GET /football-league-team?teamid=<team_id>
        Info de liga asociada a un equipo.
        """
        params = {"teamid": team_id}
        data = self._get("football-league-team", params=params)

        records = self._extract_list(data, root_path=["response"])  # ajustar si hace falta
        df = self._records_to_df(records)
        df = df.withColumn("team_id", F.lit(team_id))

        if with_partitions:
            df = self._add_partition_cols(df, exec_date)
        return df

    def get_team_logo_df(
        self,
        team_id: int,
        with_partitions: bool = True,
        exec_date: Optional[datetime.date] = None
    ) -> DataFrame:
        """
        GET /football-team-logo?teamid=<team_id>
        Info de logo del equipo.
        """
        params = {"teamid": team_id}
        data = self._get("football-team-logo", params=params)

        records = self._extract_list(data, root_path=["response"])
        df = self._records_to_df(records)
        df = df.withColumn("team_id", F.lit(team_id))

        if with_partitions:
            df = self._add_partition_cols(df, exec_date)
        return df

    # ------------------ PLAYERS ------------------

    def get_players_df(
        self,
        team_id: int,
        with_partitions: bool = True,
        exec_date: Optional[datetime.date] = None
    ) -> DataFrame:
        """
        GET /football-get-list-player?teamid=<team_id>

        Respuesta típica esperada:
        {
          "status": "success",
          "response": {
            "players": [ {...}, {...}, ... ]
          }
        }
        """
        params = {"teamid": team_id}
        data = self._get("football-get-list-player", params=params)

        players = self._extract_list(data, root_path=["response", "players"])
        df = self._records_to_df(players)
        df = df.withColumn("team_id", F.lit(team_id))

        if with_partitions:
            df = self._add_partition_cols(df, exec_date)
        return df

    def get_player_detail_df(
        self,
        player_id: int,
        with_partitions: bool = True,
        exec_date: Optional[datetime.date] = None
    ) -> DataFrame:
        """
        GET /football-get-player-detail?playerid=<player_id>
        Detalle de un jugador.
        """
        params = {"playerid": player_id}
        data = self._get("football-get-player-detail", params=params)

        records = self._extract_list(data, root_path=["response"])
        df = self._records_to_df(records)
        df = df.withColumn("player_id", F.lit(player_id))

        if with_partitions:
            df = self._add_partition_cols(df, exec_date)
        return df

    def get_player_logo_df(
        self,
        player_id: int,
        with_partitions: bool = True,
        exec_date: Optional[datetime.date] = None
    ) -> DataFrame:
        """
        GET /football-get-player-logo?playerid=<player_id>
        Logo / foto del jugador.
        """
        params = {"playerid": player_id}
        data = self._get("football-get-player-logo", params=params)

        records = self._extract_list(data, root_path=["response"])
        df = self._records_to_df(records)
        df = df.withColumn("player_id", F.lit(player_id))

        if with_partitions:
            df = self._add_partition_cols(df, exec_date)
        return df
