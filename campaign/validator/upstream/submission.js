'use strict';

const { parseLine } = require('./utils');

const submission = ({ inputDataSet, submittedDataSet, outputToFile }) => {
  function createError(message, lineNumber) {
    return new Error(`Invalid submission: ${message} at line ${lineNumber}`);
  }

  function parseInputDataSet(data) {
    const dataset = {
      numContributors: 0,
      numProjects: 0,
      contributors: {},
      projects: {},
      numTimesContributorsGotMentored: 0,
      numTimesContributorsIncreasedSkillLevel: 0,
      numContributorsWorkedOnProjects: 0,
    };
    const lines = data.split('\n');
    const [numContributors, numProjects] = parseLine(lines[0], [0, 1]);
    let currentLine = 1;

    dataset.numContributors = numContributors;
    dataset.numProjects = numProjects;

    for (let i = 0; i < numContributors; i++) {
      const [name, numSkills] = parseLine(lines[currentLine++], [1]);

      dataset.contributors[name] = {
        availableFrom: 0, waitingDays: 0, skills: {},
      };

      for (let j = 0; j < numSkills; j++) {
        const [skillName, level] = parseLine(lines[currentLine + j], [1]);

        dataset.contributors[name].skills[skillName] = level;
      }

      currentLine += numSkills;
    }

    for (let i = 0; i < numProjects; i++) {
      const [
        name, numDays, score, bestBeforeDay, numRoles,
      ] = parseLine(lines[currentLine++], [1, 2, 3, 4]);

      dataset.projects[name] = {
        numDays, score, bestBeforeDay, numRoles, skills: [],
      };

      for (let j = 0; j < numRoles; j++) {
        const [skillName, level] = parseLine(lines[currentLine + j], [1]);

        dataset.projects[name].skills.push({ name: skillName, level });
      }

      currentLine += numRoles;
    }

    return dataset;
  }

  function parseSubmittedDataSet(dataset, submittedData) {
    if (submittedData === '') {
      throw new Error('File is empty');
    }

    const lines = submittedData.split('\n');
    let currentLine = 0;
    let line = lines[currentLine++].trim();

    if (!line.length) {
      throw new Error(`Invalid submission: Line ${currentLine} is empty`);
    }

    if (isNaN(line)) {
      throw createError(`Can't parse integer value (${line})`, currentLine);
    }

    const numExecutedProjects = +line;

    if (numExecutedProjects > dataset.numProjects) {
      throw createError([
        `The number of completed projects is ${numExecutedProjects}`,
        `instead of being between 0 and ${dataset.numProjects}`,
      ].join(' '), currentLine);
    }

    for (let i = 0; i < numExecutedProjects; i++) {
      line = lines[currentLine++];

      if (typeof line === 'undefined') {
        throw createError([
          'Submission file has fewer lines than expected.',
          'Unexpected EOF (end of file)',
        ].join(' '), currentLine);
      }

      const projectName = line.trim();

      if (!projectName.length) {
        throw new Error(`Invalid submission: Line ${currentLine} is empty`);
      }

      const project = dataset.projects[projectName];

      if (!project) {
        throw createError(`Project "${projectName}" does not exist`, currentLine);
      }

      if (typeof project.receivedScore !== 'undefined') {
        throw createError(`Project ${projectName} is listed twice`, currentLine);
      }

      line = lines[currentLine++];

      if (typeof line === 'undefined') {
        throw createError([
          'Submission file has fewer lines than expected.',
          'Unexpected EOF (end of file)',
        ].join(' '), currentLine);
      }

      const contributorNames = parseLine(line);
      const numContributors = contributorNames.length;

      if (!numContributors) {
        throw new Error(`Invalid submission: Line ${currentLine} is empty`);
      }

      if (numContributors !== project.numRoles) {
        throw createError([
          `The number of listed contributors for project ${projectName}`,
          `is ${numContributors} instead of ${project.numRoles}`,
        ].join(' '), currentLine);
      }

      const duplicatedNames = contributorNames.filter((name, idx, array) =>
        array.indexOf(name) !== idx
      );

      if (duplicatedNames.length) {
        throw createError([
          `Contributor ${duplicatedNames[0]} is listed for more than one role`,
          `in project ${projectName}`,
        ].join(' '), currentLine);
      }

      const nonExistentContributor = contributorNames.find(name =>
        !dataset.contributors[name]
      );

      if (nonExistentContributor) {
        throw createError(
          `Contributor "${nonExistentContributor}" does not exist`, currentLine
        );
      }

      dataset.numContributorsWorkedOnProjects += numContributors;

      const contributorsAreAvailableFrom = Math.max(...contributorNames.map(
        name => dataset.contributors[name].availableFrom
      ));
      const endedOn = contributorsAreAvailableFrom + project.numDays;

      for (let j = 0; j < numContributors; j++) {
        const name = contributorNames[j];
        const contributor = dataset.contributors[name];
        const { name: skill, level } = project.skills[j];
        const mentored = (contributor.skills[skill] || 0) === level - 1 &&
          contributorNames.some(contributorName =>
            dataset.contributors[contributorName].skills[skill] >= level
          );

        if (contributor.skills[skill] >= level || mentored) {
          if ((contributor.skills[skill] || 0) <= level) {
            contributor.skills[skill] ??= 0;
            contributor.skills[skill]++;

            dataset.numTimesContributorsIncreasedSkillLevel++;
          }

          if (mentored) {
            dataset.numTimesContributorsGotMentored++;
          }

          contributor.waitingDays += contributorsAreAvailableFrom - contributor.availableFrom;
          contributor.availableFrom = endedOn;
          continue;
        }

        const highestSkillLevel = Math.max(...contributorNames.map(
          contributorName => dataset.contributors[contributorName].skills[skill] || 0
        ));

        throw createError([
          `Project ${projectName} cannot be completed because contributor`,
          `${name} has the ${skill} skill at level ${contributor.skills[skill] || 0},`,
          `while the ${skill} skill at level ${level} is required.`,
          `The highest ${skill} skill level among contributors on this project is`,
          `${highestSkillLevel}`,
        ].join(' '), currentLine);
      }

      const lateBy = Math.max(0, endedOn - project.bestBeforeDay);

      project.receivedScore = Math.max(0, project.score - lateBy);
    }

    if (typeof lines[currentLine] !== 'undefined' && lines[currentLine].trim().length) {
      throw createError(
        `Expected end of file, got: ${lines[currentLine]}`, currentLine + 1
      );
    }

    return dataset;
  }

  function createInsights(dataset) {
    const noColor = +outputToFile === 1;
    const yellow = value => noColor ? value : `\u001B[33m${value}\u001B[0m`;
    const formatNumber = value => yellow(value.toLocaleString('en-US'));
    const toPercentage = value => yellow(
      `${(value * 100).toString().match(/^-?([\d][^.]*)/)[0]}%`
    );

    const {
      numContributors, numProjects, contributors, projects,
      numTimesContributorsGotMentored, numTimesContributorsIncreasedSkillLevel,
      numContributorsWorkedOnProjects,
    } = dataset;
    const projectNames = Object.keys(projects);
    const contributorNames = Object.keys(contributors);
    const completedProjects = projectNames.filter(name =>
      typeof projects[name].receivedScore !== 'undefined'
    );
    const numProjectsCompleted = completedProjects.length;
    const {
      score, numProjectsBestBeforeDay, numProjectsCompletedTooLong,
    } = completedProjects.reduce((sum, name) => {
      const project = projects[name];

      sum.score += project.receivedScore;
      sum.numProjectsBestBeforeDay += project.receivedScore === project.score;
      sum.numProjectsCompletedTooLong += project.receivedScore === 0;

      return sum;
    }, { score: 0, numProjectsBestBeforeDay: 0, numProjectsCompletedTooLong: 0 });
    const averageWaitingDays = (contributorNames.reduce(
      (sum, name) => sum + contributors[name].waitingDays, 0
    ) / numContributorsWorkedOnProjects || 0).toFixed(2);
    const numContributorsWorked = contributorNames.filter(name =>
      contributors[name].availableFrom > 0
    ).length;

    return {
      score,
      toString: () => [
        `Score: ${formatNumber(score)}`,
        'Insights about this submission:',
        [
          [
            `- projects completed: ${formatNumber(numProjectsCompleted)}`,
            `(${toPercentage(numProjectsCompleted / numProjects)} of all projects)`,
          ].join(' '),
          [
            '- projects completed before their "best before" time in days',
            `(scoring full points): ${formatNumber(numProjectsBestBeforeDay)}`,
            `(${toPercentage(numProjectsBestBeforeDay / numProjects)} of all projects)`,
          ].join(' '),
          [
            '- projects completed but scoring 0 points (because they were',
            'completed too long after their "best before" time):',
            formatNumber(numProjectsCompletedTooLong),
            `(${toPercentage(numProjectsCompletedTooLong / numProjects)} of all projects)`,
          ].join(' '),
          [
            '- number of times a contributor got mentored on a project:',
            formatNumber(numTimesContributorsGotMentored),
          ].join(' '),
          [
            '- number of times a contributor increased their skill level thanks to',
            `completing a project: ${formatNumber(numTimesContributorsIncreasedSkillLevel)}`,
          ].join(' '),
          [
            '- contributors were waiting for their next project to start',
            `${formatNumber(averageWaitingDays)} days on average`,
          ].join(' '),
          [
            `- ${formatNumber(numContributorsWorked)} contributors out of`,
            `${formatNumber(numContributors)} worked on at least one project`,
          ].join(' '),
        ].join('\n'),
      ].join('\n\n'),
    };
  }

  function evaluate() {
    const dataset = parseInputDataSet(inputDataSet);

    return createInsights(parseSubmittedDataSet(dataset, submittedDataSet));
  }

  return { evaluate };
};

module.exports = submission;
