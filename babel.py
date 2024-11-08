import click
lines = open('cur.out', 'r').readlines()
@click.command()
@click.option('-thres', type=int)
def main(thres):
    ret_ls = []
    for line in lines:
        # check if it is available
        if line.split()[2].split('/')[0] == '1':
            if int(line.split()[-2].split(':')[2][0]) - int(line.split()[-1].split(':')[2][0]) > thres:
                ret_ls += [[line.split()[1],line.split()[-2],line.split()[-1]]]
    for ele in ret_ls:
        print(ele)
if __name__ == "__main__":
    main()