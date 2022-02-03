import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import pandas as pd
import random


class AllocationLog:
    """ wrapper for task allocations. this is logged by the sim routine. """

    def __init__(self, job_no, machine_no, task_start, task_duration):
        self.machine_no = machine_no
        self.job_no = job_no
        self.task_start = task_start
        self.task_duration = task_duration
        self.task_end = task_duration + task_start


def plot(title, station_logs):
    gantt_diagrams = __init_gantt_diagrams(
        sorted(
            list(map(lambda x: (x.job_no, str(x.machine_no), x.task_start, x.task_duration, x.task_end), station_logs)),
            key=lambda x: (x[1], x[2]),
        ),
        title,
    )
    rows = 2
    fig = __generate_figure(title, gantt_diagrams, rows)
    __show_formatted(fig, gantt_diagrams, rows)


def __init_gantt_diagrams(gantt_data_stations, title):
    if not gantt_data_stations:
        raise Exception("no gantt data!")

    gantts = []
    for i in range(2):
        entries = {"Task": [], "Start": [], "Finish": [], "Resource": []}
        by_machines = i == 0
        prefix_resource = "M"
        data = gantt_data_stations

        for (job, res, startTime, duration, endTime) in data:
            if by_machines:
                task = prefix_resource + str(res)
                resource = "J" + str(job)
            else:
                task = "J" + str(job)
                resource = prefix_resource + str(res)
            entries["Task"].append(task)
            entries["Resource"].append(resource)
            entries["Start"].append(startTime)
            entries["Finish"].append(endTime)
        df = pd.DataFrame(entries)
        if not by_machines:
            df = df.sort_values(by=["Finish", "Resource"], ascending=False)
        else:
            df = df.sort_values(by=["Task"])
        colors = __get_colors(150)
        gantts.append(
            ff.create_gantt(df, index_col="Resource", colors=colors, show_colorbar=True, title=title, group_tasks=True)
        )
        gantts[i].update_layout(xaxis_type="linear")
    return gantts


def __generate_figure(title, gantt_diagrams, rows):
    specs = [
        [{"type": "scatter"}],
        [{"type": "scatter"}],
    ]
    fig = make_subplots(
        rows=rows,
        cols=1,
        vertical_spacing=0.01,
        shared_xaxes=True,
        subplot_titles=(
            str(f"Machine allocations ({title})"),
            str(f"Job sequences ({title})"),
        ),
        specs=specs,
    )
    for i in range(rows):
        for trace in gantt_diagrams[i].data:
            fig.add_trace(trace, row=i + 1, col=1)

    return fig


def __show_formatted(fig, gantt_diagrams, rows):
    fig.layout.xaxis.update(gantt_diagrams[0].layout.xaxis)
    fig.layout.yaxis.update(gantt_diagrams[0].layout.yaxis)
    fig.layout.xaxis2.update(gantt_diagrams[1].layout.xaxis)
    fig.layout.yaxis2.update(gantt_diagrams[1].layout.yaxis)
    diag_height = 700
    fig.update_layout(
        height=diag_height * rows,
        margin=dict(l=20, r=20, t=20, b=20),
    )

    fig.show()


def __get_colors(number):
    r = lambda: random.randint(0, 255)
    colors = []
    colors.append("#000000")
    colors.append("#ff0000")
    colors.append("#ffff00")
    colors.append("#ff00ff")
    colors.append("#00ff00")
    colors.append("#b2b2b2")
    colors.append("#ff8c00")
    colors.append("#ff99dd")
    colors.append("#5b8f1f")
    colors.append("#00308a")
    colors.append("#82002b")
    colors.append("#5c5c5c")
    colors.append("#b8ff99")
    colors.append("#99c0ff")

    for _ in range(1, number + 1):
        colors.append("#%02X%02X%02X" % (r(), r(), r()))

    return colors